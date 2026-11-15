"""Unit tests for the pure (non-filesystem) suite versioning primitives.

Filesystem-backed behavior of ``SuiteVersionStore`` lives in
``test_suite_versioning_store.py``.
"""

from __future__ import annotations

import pytest

import great_expectations.expectations as gxe
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.suite_versioning import (
    REDACTED_PLACEHOLDER,
    MigrationPlan,
    SuiteBinding,
    SuiteCompatibilityError,
    apply_migration,
    build_migration_plan,
    canonical_json,
    check_suite_compatibility,
    content_hash,
    diff_suites,
    redact_credentials,
)
from great_expectations.exceptions import SuiteMigrationError
from great_expectations.expectations.expectation import ColumnMapExpectation
from great_expectations.expectations.registry import register_expectation

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ExpectColumnCustomMetric(ColumnMapExpectation):
    """Minimal registered custom expectation used to prove custom types round-trip."""

    expectation_type = "expect_column_custom_metric"
    map_metric = "column_values.custom_metric"
    success_keys = ("mostly",)

    def _validate(self, *args, **kwargs):  # pragma: no cover - never executed
        raise NotImplementedError


register_expectation(ExpectColumnCustomMetric)


def _suite(name: str, *expectations) -> ExpectationSuite:
    return ExpectationSuite(name=name, expectations=list(expectations))


# module level markers
pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Hashing and canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalization:
    def test_canonical_json_is_key_order_independent(self):
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_content_hash_is_stable_across_calls_and_dicts(self):
        left = {"z": [1, 2, 3], "a": {"k": "v"}}
        right = {"a": {"k": "v"}, "z": [1, 2, 3]}
        assert content_hash(left) == content_hash(right)
        assert content_hash(left) != content_hash({**left, "z": [1, 2, 4]})
        assert len(content_hash(left)) == 64  # sha256 hexdigest


# ---------------------------------------------------------------------------
# Credential redaction
# ---------------------------------------------------------------------------


class TestRedactCredentials:
    def test_redacts_sensitive_keys_at_any_depth(self):
        payload = {
            "outer": {"api_key": "abc123", "nested": {"password": "hunter2", "ok": "kept"}},
            "list": [{"token": "tkn"}],
        }
        redacted = redact_credentials(payload)
        assert redacted["outer"]["api_key"] == REDACTED_PLACEHOLDER
        assert redacted["outer"]["nested"]["password"] == REDACTED_PLACEHOLDER
        assert redacted["outer"]["nested"]["ok"] == "kept"
        assert redacted["list"][0]["token"] == REDACTED_PLACEHOLDER
        # the input is never mutated
        assert payload["outer"]["api_key"] == "abc123"

    def test_redacts_embedded_connection_urls_and_kv_strings(self):
        assert (
            redact_credentials("postgresql://admin:supersecret@host:5432/db")
            == REDACTED_PLACEHOLDER
        )
        assert redact_credentials("pwd=topsecret;other=1") == REDACTED_PLACEHOLDER
        assert redact_credentials("nothing sensitive here") == "nothing sensitive here"

    def test_non_sensitive_content_is_untouched(self):
        payload = {"column": "users", "value_set": [1, 2, 3], "n": None}
        assert redact_credentials(payload) == payload


# ---------------------------------------------------------------------------
# Field-level diff
# ---------------------------------------------------------------------------


class TestDiffSuites:
    def test_identical_suites_have_no_changes(self):
        suite = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="a"))
        diff = diff_suites(suite, suite)
        assert diff.has_changes is False
        assert diff.expectation_changes == ()
        assert diff.column_changes == ()

    def test_detects_added_and_removed_expectations(self):
        old = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="a"))
        new = _suite(
            "s",
            gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=10),
        )
        diff = diff_suites(old, new)
        ops = {change.op for change in diff.expectation_changes}
        assert ops == {"add", "remove"}
        columns = {change.column: change.change_type for change in diff.column_changes}
        assert columns == {"a": "removed", "b": "added"}

    def test_detects_kwarg_level_change_on_the_same_expectation(self):
        old = _suite("s", gxe.ExpectColumnValuesToBeInSet(column="status", value_set=["a", "b"]))
        new = _suite(
            "s",
            gxe.ExpectColumnValuesToBeInSet(column="status", value_set=["a", "b", "c"]),
        )
        diff = diff_suites(old, new)
        assert diff.has_changes is True
        # the expectation itself is matched (same domain), only a field under it changed
        assert all("/expectations/" in change.path for change in diff.expectation_changes)
        assert any("value_set" in change.path for change in diff.expectation_changes)

    def test_classifies_column_rename_instead_of_add_remove_pair(self):
        old = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="old_name"))
        new = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="new_name"))
        diff = diff_suites(old, new)
        column_changes = {(c.column, c.change_type) for c in diff.column_changes}
        assert column_changes == {("old_name -> new_name", "renamed")}

    def test_detects_column_type_change(self):
        old = _suite("s", gxe.ExpectColumnValuesToBeOfType(column="age", type_="int64"))
        new = _suite("s", gxe.ExpectColumnValuesToBeOfType(column="age", type_="float64"))
        diff = diff_suites(old, new)
        type_changes = [c for c in diff.column_changes if c.change_type == "type_changed"]
        assert len(type_changes) == 1
        assert type_changes[0].details[0].old == "int64"
        assert type_changes[0].details[0].new == "float64"

    def test_detects_suite_level_field_changes(self):
        old = ExpectationSuite(name="s", suite_parameters={"threshold": 10})
        new = ExpectationSuite(name="s", suite_parameters={"threshold": 20})
        diff = diff_suites(old, new)
        paths = {change.path for change in diff.field_changes}
        assert "/suite_parameters" in paths

    def test_diff_to_dict_round_trips(self):
        old = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="a"))
        new = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="b"))
        payload = diff_suites(old, new, 1, 2).to_dict()
        assert payload["version_from"] == 1
        assert payload["version_to"] == 2
        assert payload["has_changes"] is True
        assert payload["column_changes"][0]["change_type"] in {"renamed", "added", "removed"}


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------


class TestCheckCompatibility:
    def test_removed_column_is_breaking(self):
        old = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="a"))
        new = ExpectationSuite(name="s")
        report = check_suite_compatibility(old, new)
        assert report.compatible is False
        assert any(issue.code == "COLUMN_REMOVED" for issue in report.issues)

    def test_type_change_is_breaking(self):
        old = _suite("s", gxe.ExpectColumnValuesToBeOfType(column="age", type_="int64"))
        new = _suite("s", gxe.ExpectColumnValuesToBeOfType(column="age", type_="float64"))
        report = check_suite_compatibility(old, new)
        assert report.compatible is False
        assert any(issue.code == "COLUMN_TYPE_CHANGED" for issue in report.issues)

    def test_unregistered_custom_expectation_is_a_warning_not_blocking(self):
        # Unregistered types are stored as raw dicts (the version store never drops them);
        # compatibility is checked against that dict representation.
        old = {"name": "s", "expectations": []}
        new = {
            "name": "s",
            "expectations": [
                {
                    "type": "expect_totally_unknown_custom",
                    "kwargs": {"column": "z"},
                    "meta": {},
                    "severity": "critical",
                }
            ],
        }
        report = check_suite_compatibility(old, new)
        custom_warnings = [
            issue for issue in report.issues if issue.code == "CUSTOM_EXPECTATION_UNREGISTERED"
        ]
        assert len(custom_warnings) == 1
        assert custom_warnings[0].severity == "warning"
        # additions alone, even unknown custom ones, do not break the old suite
        assert report.compatible is True

    def test_registered_custom_expectation_produces_no_warning(self):
        old = ExpectationSuite(name="s")
        new = _suite("s", ExpectColumnCustomMetric(column="c"))
        report = check_suite_compatibility(old, new)
        assert all(issue.code != "CUSTOM_EXPECTATION_UNREGISTERED" for issue in report.issues)

    def test_expected_column_type_mismatch_is_breaking(self):
        new = _suite("s", gxe.ExpectColumnValuesToBeOfType(column="age", type_="int64"))
        report = check_suite_compatibility(new, new, expected_column_types={"age": "float64"})
        assert report.compatible is False
        assert any(issue.code == "COLUMN_TYPE_MISMATCH" for issue in report.issues)

    def test_compatibility_report_serializes(self):
        old = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="a"))
        payload = check_suite_compatibility(old, ExpectationSuite(name="s")).to_dict()
        assert payload["compatible"] is False
        assert payload["issues"][0]["severity"] == "breaking"

    def test_suite_compatibility_error_carries_code_and_path(self):
        error = SuiteCompatibilityError("X", "msg", path="/a")
        assert error.code == "X"
        assert error.path == "/a"
        assert "[X] msg" in str(error)


# ---------------------------------------------------------------------------
# Migration plans and application
# ---------------------------------------------------------------------------


class TestMigrationPlan:
    def test_plan_inferred_from_diff_detects_rename_drop_and_type(self):
        old = _suite(
            "s",
            gxe.ExpectColumnValuesToBeInSet(column="old_name", value_set=[1]),
            gxe.ExpectColumnValuesToBeOfType(column="age", type_="int64"),
            gxe.ExpectColumnValuesToNotBeNull(column="doomed"),
        )
        new = _suite(
            "s",
            gxe.ExpectColumnValuesToBeInSet(column="new_name", value_set=[1]),
            gxe.ExpectColumnValuesToBeOfType(column="age", type_="float64"),
        )
        plan = build_migration_plan(old, new)
        # the rename is unambiguous: only one column carries that expectation-type set
        assert plan.renamed_columns == {"old_name": "new_name"}
        assert "doomed" in plan.dropped_columns
        assert plan.type_changes == {"age": ("int64", "float64")}
        # plan is serializable for audit trails
        payload = plan.to_dict()
        assert payload["renamed_columns"] == {"old_name": "new_name"}

    def test_ambiguous_same_type_columns_are_not_guessed_as_renames(self):
        # two old columns with identical type sets: a rename cannot be inferred unambiguously,
        # so the diff conservatively reports an add plus removals instead of guessing
        old = _suite(
            "s",
            gxe.ExpectColumnValuesToNotBeNull(column="old_one"),
            gxe.ExpectColumnValuesToNotBeNull(column="old_two"),
        )
        new = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="new_one"))
        plan = build_migration_plan(old, new)
        assert plan.renamed_columns == {}
        assert plan.dropped_columns == frozenset({"old_one", "old_two"})

    def test_binding_matches_on_flat_and_nested_scope_meta(self):
        binding = SuiteBinding(datasource_name="pg", table_name="users")
        flat = {"meta": {"datasource_name": "pg", "table_name": "users"}}
        nested = {"meta": {"scope": {"datasource_name": "pg", "table_name": "users"}}}
        other = {"meta": {"datasource_name": "snow", "table_name": "orders"}}
        assert binding.matches(flat) is True
        assert binding.matches(nested) is True
        assert binding.matches(other) is False
        assert SuiteBinding().matches(other) is True  # unscoped matches everything


class TestApplyMigration:
    def _scoped_suite(self) -> ExpectationSuite:
        return _suite(
            "s",
            gxe.ExpectColumnValuesToNotBeNull(
                column="old_name",
                meta={"scope": {"datasource_name": "pg", "table_name": "users"}},
            ),
            gxe.ExpectColumnValuesToNotBeNull(
                column="old_name",
                meta={"scope": {"datasource_name": "snow", "table_name": "orders"}},
            ),
        )

    def test_rename_respects_binding_scope(self):
        suite = self._scoped_suite()
        plan = MigrationPlan(
            renamed_columns={"old_name": "new_name"},
            dropped_columns=frozenset(),
            type_changes={},
            scopes=(SuiteBinding(datasource_name="pg", table_name="users"),),
        )
        result = apply_migration(suite, plan, strict=False)
        columns = {
            expectation.configuration.kwargs["column"]
            for expectation in result.migrated_suite.expectations
        }
        assert columns == {"new_name", "old_name"}

    def test_dropped_column_removes_only_in_scope_expectations(self):
        suite = _suite(
            "s",
            gxe.ExpectColumnValuesToNotBeNull(
                column="doomed",
                meta={"datasource_name": "pg", "table_name": "users"},
            ),
            gxe.ExpectColumnValuesToNotBeNull(column="doomed"),
        )
        plan = MigrationPlan(
            renamed_columns={},
            dropped_columns=frozenset({"doomed"}),
            type_changes={},
            scopes=(SuiteBinding(datasource_name="pg", table_name="users"),),
        )
        result = apply_migration(suite, plan, strict=False)
        assert len(result.migrated_suite.expectations) == 1
        assert result.removed_expectation_types == ("expect_column_values_to_not_be_null",)

    def test_type_change_updates_type_kwarg(self):
        suite = _suite("s", gxe.ExpectColumnValuesToBeOfType(column="age", type_="int64"))
        plan = MigrationPlan(
            renamed_columns={},
            dropped_columns=frozenset(),
            type_changes={"age": ("int64", "float64")},
            scopes=(),
        )
        result = apply_migration(suite, plan, strict=False)
        kwargs = result.migrated_suite.expectations[0].configuration.kwargs
        assert kwargs["type_"] == "float64"
        assert "expect_column_values_to_be_of_type" in result.affected_expectation_types

    def test_strict_mode_refuses_breaking_migration_and_returns_nothing(self):
        old = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="doomed"))
        plan = MigrationPlan(
            renamed_columns={},
            dropped_columns=frozenset({"doomed"}),
            type_changes={},
            scopes=(),
        )
        with pytest.raises(SuiteMigrationError, match="incompatible migration"):
            apply_migration(old, plan, strict=True)
        # the original suite object is untouched - no half migration
        assert old.expectations[0].configuration.kwargs["column"] == "doomed"

    def test_non_strict_migration_reports_breaking_issues(self):
        old = _suite("s", gxe.ExpectColumnValuesToNotBeNull(column="doomed"))
        plan = MigrationPlan(
            renamed_columns={},
            dropped_columns=frozenset({"doomed"}),
            type_changes={},
            scopes=(),
        )
        result = apply_migration(old, plan, strict=False)
        assert result.compatibility.compatible is False
        assert any(issue.code == "EXPECTATION_REMOVED" for issue in result.compatibility.issues)
        assert len(result.migrated_suite.expectations) == 0
