"""Filesystem-backed tests for the immutable suite version store."""

from __future__ import annotations

import json
import pathlib
import threading

import pytest

import great_expectations.expectations as gxe
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.suite_versioning import (
    DEFAULT_MAX_VERSIONS,
    MigrationPlan,
    SuiteMigrator,
    SuiteVersionStore,
)
from great_expectations.exceptions import (
    SuiteMigrationError,
    SuiteVersionConflictError,
    SuiteVersionCorruptError,
    SuiteVersionNotFoundError,
)

# module level markers
pytestmark = [pytest.mark.unit, pytest.mark.filesystem]


def _suite(name: str, *columns: str) -> ExpectationSuite:
    return ExpectationSuite(
        name=name,
        expectations=[gxe.ExpectColumnValuesToNotBeNull(column=column) for column in columns],
    )


@pytest.fixture
def store(tmp_path: pathlib.Path) -> SuiteVersionStore:
    return SuiteVersionStore(tmp_path)


class TestCommitAndRoundTrip:
    def test_first_commit_is_version_one_and_immutable(self, store: SuiteVersionStore):
        suite = _suite("s", "a")
        version = store.commit(suite, message="initial", author="alice", tags=["r1"])
        assert version.version == 1
        assert version.parent_version is None
        assert version.author == "alice"
        assert version.tags == ("r1",)
        assert version.content_hash
        # committing the same suite again creates a new record; v1 stays reachable
        store.commit(_suite("s", "a", "b"))
        assert store.load_suite("s", 1) == suite

    def test_round_trip_reconstructs_every_committed_version(self, store: SuiteVersionStore):
        suites = []
        for index in range(1, 6):
            suite = _suite("s", *(f"col_{i}" for i in range(index)))
            store.commit(suite, message=f"v{index}")
            suites.append(suite)
        for index, suite in enumerate(suites, start=1):
            assert store.load_suite("s", index) == suite
        assert store.load_suite("s") == suites[-1]  # default is head

    def test_versions_are_listed_in_commit_order_with_metadata(self, store: SuiteVersionStore):
        store.commit(_suite("s", "a"))
        store.commit(_suite("s", "a", "b"))
        versions = store.list_versions("s")
        assert [version.version for version in versions] == [1, 2]
        assert versions[1].parent_version == 1
        assert versions[0].created_at <= versions[1].created_at
        assert store.head_version("s").version == 2

    def test_loading_unknown_version_or_suite_raises(self, store: SuiteVersionStore):
        store.commit(_suite("s", "a"))
        with pytest.raises(SuiteVersionNotFoundError):
            store.load_suite("s", 99)
        with pytest.raises(SuiteVersionNotFoundError):
            store.load_suite("never_committed")

    def test_multiple_suites_are_stored_under_isolated_directories(self, tmp_path: pathlib.Path):
        store = SuiteVersionStore(tmp_path)
        store.commit(_suite("suite one", "a"))
        store.commit(_suite("suite/two", "b"))
        entries = {path.name for path in tmp_path.iterdir()}
        assert "suite__one" in entries  # whitespace slugified
        assert "suite__two" in entries  # slash slugified
        assert store.load_suite("suite one") == _suite("suite one", "a")
        assert store.load_suite("suite/two") == _suite("suite/two", "b")


class TestDeltaStorage:
    def test_store_keeps_deltas_not_full_copies(self, tmp_path: pathlib.Path):
        # a two-entry cache (oldest anchor + head) so intermediate snapshots get evicted
        store = SuiteVersionStore(tmp_path, max_versions=10, snapshot_cache_size=2)
        store.commit(_suite("s", "a"))
        for index in range(2, 8):
            store.commit(_suite("s", "a", f"b{index}"))
        log_lines = (tmp_path / "s" / "versions.jsonl").read_text().strip().splitlines()
        snapshot_files = list((tmp_path / "s" / "snapshots").rglob("v*.json"))
        # the bounded cache holds far fewer snapshots than the seven commits...
        assert len(snapshot_files) <= 2
        # ...while the log records a reverse delta per non-anchor version
        delta_records = [
            json.loads(line) for line in log_lines if "reverse_patch" in json.loads(line)
        ]
        assert len(delta_records) >= 4
        # every retained version still reconstructs despite the small snapshot cache
        for version in range(1, 8):
            assert store.load_suite("s", version) is not None

    def test_reconstructed_payload_fails_closed_on_tampered_content(self, tmp_path: pathlib.Path):
        # cache_size=2 keeps the oldest anchor and head; an intermediate version is stored
        # only as a reverse delta, which gives us a delta record to tamper with.
        store = SuiteVersionStore(tmp_path, max_versions=10, snapshot_cache_size=2)
        store.commit(_suite("s", "a"))
        store.commit(_suite("s", "a", "b"))
        store.commit(_suite("s", "a", "b", "c"))
        log_path = store._log_path("s")
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        delta = next(record for record in records if "reverse_patch" in record)
        assert delta["version"] >= 2

        # (1) a structurally invalid delta is rejected while applying, fail-closed.
        # v2 is rebuilt by reverse-applying its successor's (v3) delta, so corrupt that one.
        successor = next(record for record in records if record["version"] == delta["version"] + 1)
        successor["reverse_patch"] = [
            {"op": "remove", "path": "/expectations/99"}  # index out of range
        ]
        log_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        with pytest.raises(SuiteVersionCorruptError):
            store.load_suite("s", delta["version"])

        # (2) a delta that still applies but yields the wrong content fails the hash check
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        successor = next(record for record in records if record["version"] == delta["version"] + 1)
        successor["reverse_patch"] = [
            {"op": "replace", "path": "/name", "value": "forged-suite-name"}
        ]
        log_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        with pytest.raises(SuiteVersionCorruptError, match="integrity check"):
            store.load_suite("s", delta["version"])


class TestRetentionBounds:
    def test_max_versions_bounds_the_log(self, tmp_path: pathlib.Path):
        store = SuiteVersionStore(tmp_path, max_versions=5, snapshot_cache_size=2)
        suites = []
        for index in range(1, 9):
            suite = _suite("s", f"col_{index}")
            store.commit(suite)
            suites.append(suite)
        versions = store.list_versions("s")
        assert [version.version for version in versions] == [4, 5, 6, 7, 8]
        # pruned versions are gone, retained ones all reconstruct exactly
        with pytest.raises(SuiteVersionNotFoundError):
            store.load_suite("s", 1)
        for version in versions:
            assert store.load_suite("s", version.version) == suites[version.version - 1]

    def test_snapshot_cache_is_bounded(self, tmp_path: pathlib.Path):
        store = SuiteVersionStore(tmp_path, max_versions=50, snapshot_cache_size=3)
        for index in range(10):
            store.commit(_suite("s", f"col_{index}"))
        snapshot_files = list((tmp_path / "s" / "snapshots").rglob("v*.json"))
        assert len(snapshot_files) <= 3

    def test_invalid_bounds_are_rejected(self, tmp_path: pathlib.Path):
        with pytest.raises(ValueError):
            SuiteVersionStore(tmp_path, max_versions=0)
        with pytest.raises(ValueError):
            SuiteVersionStore(tmp_path, snapshot_cache_size=1)

    def test_defaults_are_finite(self):
        assert DEFAULT_MAX_VERSIONS >= 1


class TestDiffAndRollback:
    def test_diff_between_committed_versions(self, store: SuiteVersionStore):
        store.commit(_suite("s", "a"))
        store.commit(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToBeInSet(column="status", value_set=["x", "y"])
                ],
            )
        )
        diff = store.diff("s", 1, 2)
        assert diff.version_from == 1
        assert diff.version_to == 2
        assert diff.has_changes is True
        column_changes = {change.column: change.change_type for change in diff.column_changes}
        assert column_changes == {"a": "removed", "status": "added"}

    def test_rollback_creates_a_new_immutable_version_with_old_content(
        self, store: SuiteVersionStore
    ):
        first = _suite("s", "a")
        store.commit(first)
        store.commit(_suite("s", "a", "b"))
        head_before = store.head_version("s").version
        rollback_version = store.rollback("s", 1, message="undo")
        # history is appended to, never rewritten
        assert rollback_version.version == head_before + 1
        assert rollback_version.tags == ("rollback",)
        assert rollback_version.message == "undo"
        assert store.load_suite("s") == first
        # the rollback target is still reachable
        assert store.load_suite("s", 1) == first

    def test_rollback_to_unknown_version_raises(self, store: SuiteVersionStore):
        store.commit(_suite("s", "a"))
        with pytest.raises(SuiteVersionNotFoundError):
            store.rollback("s", 42)


class TestCorruption:
    def test_corrupt_log_line_raises_and_keeps_other_suites_intact(self, store: SuiteVersionStore):
        store.commit(_suite("s", "a"))
        with open(store._log_path("s"), "a") as handle:
            handle.write("{not valid json\n")
        with pytest.raises(SuiteVersionCorruptError, match="Corrupt version log"):
            store.list_versions("s")
        # a different suite's store is unaffected
        store.commit(_suite("other", "z"))
        assert store.load_suite("other") == _suite("other", "z")

    def test_corrupt_snapshot_is_detected_then_rebuilt_by_repair(self, store: SuiteVersionStore):
        store.commit(_suite("s", "a"))
        store.commit(_suite("s", "a", "b"))
        store.commit(_suite("s", "a", "b", "c"))
        first_snapshot = store._snapshot_path("s", 1)
        first_snapshot.write_text("{broken")
        with pytest.raises(SuiteVersionCorruptError):
            store.load_suite("s", 1)
        # a dry run reports the corrupt snapshot without changing anything
        corrupt = store.repair("s")
        assert len(corrupt) == 1
        assert first_snapshot.read_text() == "{broken"
        # the real repair reports what it found and rebuilds the snapshot in place
        repaired = store.repair("s", dry_run=False)
        assert len(repaired) == 1
        # a subsequent scan finds nothing corrupt and every version reconstructs
        assert store.repair("s") == []
        assert store.load_suite("s", 1) == _suite("s", "a")

    def test_malformed_record_without_hash_is_rejected(self, store: SuiteVersionStore):
        store.commit(_suite("s", "a"))
        with open(store._log_path("s"), "a") as handle:
            handle.write(json.dumps({"version": 2}) + "\n")
        with pytest.raises(SuiteVersionCorruptError, match="Malformed version record"):
            store.list_versions("s")


class TestCredentials:
    def test_credentials_are_never_written_to_log_or_snapshots(self, tmp_path: pathlib.Path):
        store = SuiteVersionStore(tmp_path)
        suite = ExpectationSuite(
            name="secret",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a", meta={"password": "hunter2"})
            ],
            meta={"db": "postgresql://admin:supersecret@host/db"},
        )
        store.commit(suite, message="pw=topsecret", author="bob")
        log_text = (tmp_path / "secret" / "versions.jsonl").read_text()
        snapshot_text = next((tmp_path / "secret" / "snapshots").rglob("v*.json")).read_text()
        for secret in ("hunter2", "supersecret", "topsecret"):
            assert secret not in log_text
            assert secret not in snapshot_text

    def test_nested_credential_keys_in_tags_and_meta_are_redacted(self, store: SuiteVersionStore):
        suite = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a", meta={"conn_str": "amqp://u:p@host"})
            ],
        )
        store.commit(suite, tags=["token=abcdef"])
        raw = store._log_path("s").read_text()
        assert "abcdef" not in raw
        assert "u:p@host" not in raw


class TestConcurrency:
    def test_stale_expected_parent_raises_conflict(self, store: SuiteVersionStore):
        suite = _suite("s", "a")
        store.commit(suite)
        with pytest.raises(SuiteVersionConflictError):
            store.commit(suite, expected_parent_version=99)

    def test_concurrent_commits_serialize_without_losing_versions(self, store: SuiteVersionStore):
        store.commit(_suite("s", "base"))
        thread_count = 16
        outcomes: list[str] = []

        def worker(index: int) -> None:
            for _ in range(thread_count):
                try:
                    head = store.head_version("s").version
                    store.commit(
                        _suite("s", "base", f"worker_{index}"),
                        expected_parent_version=head,
                    )
                    outcomes.append("ok")
                    return
                except SuiteVersionConflictError:
                    continue
            outcomes.append("failed")

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert outcomes.count("ok") == thread_count
        assert "failed" not in outcomes
        versions = store.list_versions("s")
        assert len(versions) == thread_count + 1
        # full chain integrity: every committed version reconstructs and verifies its hash
        for version in versions:
            store.load_suite("s", version.version)

    def test_failed_migration_leaves_store_untouched(self, store: SuiteVersionStore):
        suite = _suite("s", "doomed")
        store.commit(suite)
        migrator = SuiteMigrator(store)
        plan = MigrationPlan(
            renamed_columns={},
            dropped_columns=frozenset({"doomed"}),
            type_changes={},
            scopes=(),  # unscoped: applies to every expectation in the suite
        )
        with pytest.raises(SuiteMigrationError, match="incompatible migration"):
            migrator.migrate(suite, plan, strict=True)
        # no new version was written and the old content is intact
        assert len(store.list_versions("s")) == 1
        assert store.load_suite("s") == suite
