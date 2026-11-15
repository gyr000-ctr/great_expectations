"""End-to-end integration tests for suite versioning and migration.

These exercise the full Great Expectations workflow - building a suite against a real pandas
data source, committing immutable versions, migrating the suite when the underlying table
changes (column rename, drop and type change), validating against real data, and rolling back -
rather than relying solely on mocked components. No external database or cloud service is
required; the only external surface is the local filesystem used by the version store.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.suite_versioning import (
    MigrationPlan,
    SuiteBinding,
    SuiteMigrator,
    SuiteVersionStore,
)
from great_expectations.exceptions import SuiteMigrationError

# module level markers
pytestmark = [pytest.mark.integration, pytest.mark.filesystem]


def _pandas_batch(context, dataframe: pd.DataFrame, name: str):
    datasource = context.data_sources.add_pandas(name=f"pandas_ds_{name}")
    asset = datasource.add_dataframe_asset(name=f"asset_{name}")
    batch_definition = asset.add_batch_definition_whole_dataframe(name=f"whole_frame_{name}")
    return batch_definition.get_batch(batch_parameters={"dataframe": dataframe})


def _initial_suite() -> ExpectationSuite:
    return ExpectationSuite(
        name="users_suite",
        expectations=[
            gxe.ExpectColumnValuesToNotBeNull(
                column="user_name",
                meta={"scope": {"datasource_name": "warehouse", "table_name": "users"}},
            ),
            gxe.ExpectColumnValuesToBeInSet(
                column="status",
                value_set=["active", "inactive"],
                meta={"scope": {"datasource_name": "warehouse", "table_name": "users"}},
            ),
            gxe.ExpectColumnValuesToBeOfType(
                column="age",
                type_="int64",
                meta={"scope": {"datasource_name": "warehouse", "table_name": "users"}},
            ),
        ],
    )


class TestSuiteVersioningMigrationEndToEnd:
    def test_commit_migrate_validate_and_rollback_lifecycle(self, tmp_path: pathlib.Path):
        context = gx.get_context(mode="ephemeral")
        # The original table: a user_name column, a status column and integer ages.
        original_frame = pd.DataFrame(
            {
                "user_name": ["alice", "bob"],
                "status": ["active", "inactive"],
                "age": pd.Series([30, 41], dtype="int64"),
            }
        )
        original_batch = _pandas_batch(context, original_frame, name="original")

        suite = _initial_suite()
        initial_result = original_batch.validate(suite)
        assert initial_result.success

        # The current, un-versioned behavior is unchanged until a store is explicitly used.
        store = SuiteVersionStore(tmp_path / "versions")
        migrator = SuiteMigrator(store)
        first = store.commit(suite, message="baseline before schema change")
        assert first.version == 1

        # The table evolves: user_name -> username, status is dropped, age becomes float.
        migrated_frame = pd.DataFrame(
            {
                "username": ["alice", "bob"],
                "age": pd.Series([30.0, 41.5], dtype="float64"),
            }
        )
        migrated_batch = _pandas_batch(context, migrated_frame, name="migrated")
        plan = MigrationPlan(
            renamed_columns={"user_name": "username"},
            dropped_columns=frozenset({"status"}),
            type_changes={"age": ("int64", "float64")},
            scopes=(SuiteBinding(datasource_name="warehouse", table_name="users"),),
        )

        # Strict mode refuses the breaking drop; nothing is committed on failure.
        with pytest.raises(SuiteMigrationError):
            migrator.migrate(suite, plan, strict=True)
        assert [v.version for v in store.list_versions("users_suite")] == [1]

        # A scoped, non-strict migration is recorded as a new immutable version.
        result, version = migrator.migrate(suite, plan, strict=False)
        assert version.version == 2
        migrated_suite = result.migrated_suite
        assert "status" not in {
            expectation.configuration.kwargs.get("column")
            for expectation in migrated_suite.expectations
        }
        # The migrated suite validates successfully against the evolved table.
        migrated_validation = migrated_batch.validate(migrated_suite)
        assert migrated_validation.success

        # The field-level diff documents exactly what changed between the two versions.
        diff = store.diff("users_suite", 1, 2)
        changed_columns = {change.column for change in diff.column_changes}
        assert any("user_name -> username" in column for column in changed_columns)
        assert "status" in changed_columns

        # Rolling back restores the baseline content as a new auditable version and it still
        # validates against the original table.
        rolled_back = migrator.rollback("users_suite", 1)
        assert rolled_back == suite
        rollback_validation = original_batch.validate(rolled_back)
        assert rollback_validation.success
        assert store.head_version("users_suite").version == 3

    def test_migration_scoped_to_one_datasource_leaves_other_untouched(
        self, tmp_path: pathlib.Path
    ):
        context = gx.get_context(mode="ephemeral")
        suite = ExpectationSuite(
            name="multi_scope_suite",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(
                    column="legacy_col",
                    meta={
                        "scope": {
                            "datasource_name": "warehouse",
                            "table_name": "users",
                        }
                    },
                ),
                gxe.ExpectColumnValuesToNotBeNull(
                    column="legacy_col",
                    meta={
                        "scope": {
                            "datasource_name": "warehouse",
                            "table_name": "orders",
                        }
                    },
                ),
            ],
        )
        store = SuiteVersionStore(tmp_path / "versions")
        migrator = SuiteMigrator(store)
        store.commit(suite)
        plan = MigrationPlan(
            renamed_columns={"legacy_col": "modern_col"},
            dropped_columns=frozenset(),
            type_changes={},
            scopes=(SuiteBinding(datasource_name="warehouse", table_name="users"),),
        )
        result, _ = migrator.migrate(suite, plan, strict=False)
        columns = {
            (
                expectation.configuration.kwargs["column"],
                expectation.configuration.meta["scope"]["table_name"],
            )
            for expectation in result.migrated_suite.expectations
        }
        assert ("modern_col", "users") in columns
        assert ("legacy_col", "orders") in columns

        # The migrated suite still runs end to end against matching data.
        frame = pd.DataFrame({"modern_col": [1, 2], "legacy_col": [3, 4]})
        batch = _pandas_batch(context, frame, name="scoped")
        validation = batch.validate(result.migrated_suite)
        assert validation.success
