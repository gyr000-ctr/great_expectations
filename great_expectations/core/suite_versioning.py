"""Auditable versioning, field-level diffing, compatibility checks and migration for
Expectation Suites.

This module is opt-in: an :class:`ExpectationSuite` behaves exactly as before unless a caller
explicitly puts it under a :class:`SuiteVersionStore`. Nothing here is wired into the default
suite save path.

Design properties
------------------
* Versions are **immutable**: every commit receives an incrementing number and a content hash;
  existing version records are never rewritten.
* The store keeps an append-only reverse-delta log plus a bounded snapshot cache, so the total
  storage is ``O(number of versions * delta size + snapshot_cache_size * suite size)`` rather
  than one full copy of the suite per version.
* Writes are atomic (temp file + ``os.replace``) and guarded by a cross-process file lock;
  optimistic concurrency is available through ``expected_parent_version``. A failed migration
  never leaves a half-applied version behind.
* Anything that looks like a connection credential (in stored content or metadata) is replaced
  with a redaction placeholder before it touches disk.
* Corrupt records raise :class:`SuiteVersionCorruptError` instead of being silently repaired or
  overwritten, so the last known-good version stays available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from great_expectations.exceptions import (
    ExpectationNotFoundError,
    SuiteMigrationError,
    SuiteVersionConflictError,
    SuiteVersionCorruptError,
    SuiteVersionError,
    SuiteVersionNotFoundError,
)

if TYPE_CHECKING:
    from great_expectations.compatibility.typing_extensions import Self
    from great_expectations.core.expectation_suite import ExpectationSuite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and tunables
# ---------------------------------------------------------------------------

#: Placeholder substituted for any value that looks like a connection credential.
REDACTED_PLACEHOLDER = "***REDACTED***"

#: Metadata keys whose values are always treated as credentials.
SENSITIVE_KEY_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
    "connection_string",
    "conn_string",
    "conn_str",
    "dsn",
)

# Connection strings / URLs carrying embedded credentials, e.g.
# ``postgresql://user:pw@host/db`` or ``amqp://user:pw@host``.
_CREDENTIAL_URL_RE = re.compile(r"[a-z][a-z0-9+.\-]*://[^\s/@:]+:[^\s/@]+@", re.IGNORECASE)
# Assignments such as ``password=hunter2`` inside keyword-style connection strings.
_CREDENTIAL_KV_RE = re.compile(
    r"(?i)(password|passwd|pwd|pw|secret|token|api[_-]?key|access[_-]?key)\s*[:=]\s*([^\s;,'\"]+)"
)
_LOG_FILE_NAME = "versions.jsonl"
_LOCK_FILE_NAME = ".lock"
_SNAPSHOTS_DIR_NAME = "snapshots"

#: Default hard cap on retained versions (bounds the metadata/delta log size).
DEFAULT_MAX_VERSIONS = 100
#: Default number of full-suite snapshots kept in the bounded snapshot cache.
DEFAULT_SNAPSHOT_CACHE_SIZE = 3
#: Smallest viable snapshot cache: one anchor for the oldest retained version plus the head.
MIN_SNAPSHOT_CACHE_SIZE = 2
#: Number of stored versions per directory shard.
_SHARD_SIZE = 1000

# Patch operation kinds, following RFC 6902 naming.
_OP_ADD = "add"
_OP_REMOVE = "remove"
_OP_REPLACE = "replace"
_OP_MOVE = "move"


# ---------------------------------------------------------------------------
# Exceptions specific to the diff/compatibility value objects
# ---------------------------------------------------------------------------


class SuiteCompatibilityError(SuiteVersionError):
    """A breaking compatibility problem found when comparing two suite versions."""

    def __init__(self, code: str, message: str, path: Optional[str] = None) -> None:
        self.code = code
        self.path = path
        super().__init__(f"[{code}] {message}")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldChange:
    """A single field-level change between two suites."""

    path: str
    op: str
    old: Any = None
    new: Any = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "op": self.op,
            "old": self.old,
            "new": self.new,
        }

    @classmethod
    def from_dict(cls, value: dict) -> FieldChange:
        return cls(path=value["path"], op=value["op"], old=value.get("old"), new=value.get("new"))


@dataclass(frozen=True)
class ColumnChange:
    """A change affecting one logical column across the expectations of a suite."""

    column: str
    change_type: str
    details: tuple[FieldChange, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "change_type": self.change_type,
            "details": [change.to_dict() for change in self.details],
        }

    @classmethod
    def from_dict(cls, value: dict) -> ColumnChange:
        return cls(
            column=value["column"],
            change_type=value["change_type"],
            details=tuple(FieldChange.from_dict(item) for item in value.get("details", [])),
        )


@dataclass(frozen=True)
class SuiteDiff:
    """Field-level diff between two expectation suites."""

    version_from: Optional[int]
    version_to: Optional[int]
    expectation_changes: tuple[FieldChange, ...]
    column_changes: tuple[ColumnChange, ...]
    field_changes: tuple[FieldChange, ...]
    has_changes: bool

    def to_dict(self) -> dict:
        return {
            "version_from": self.version_from,
            "version_to": self.version_to,
            "expectation_changes": [change.to_dict() for change in self.expectation_changes],
            "column_changes": [change.to_dict() for change in self.column_changes],
            "field_changes": [change.to_dict() for change in self.field_changes],
            "has_changes": self.has_changes,
        }


@dataclass(frozen=True)
class CompatibilityIssue:
    """One finding of :func:`check_suite_compatibility`."""

    severity: str  # "breaking" or "warning"
    code: str
    message: str
    path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": str(self.code),
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of a compatibility check between two suite versions."""

    compatible: bool
    issues: tuple[CompatibilityIssue, ...]

    def to_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class SuiteVersion:
    """Metadata describing one immutable suite version.

    The full suite payload is deliberately absent: it is reconstructed from the delta log and
    bounded snapshot cache, never duplicated per version record.
    """

    version: int
    created_at: str
    parent_version: Optional[int]
    content_hash: str
    message: str
    author: Optional[str]
    tags: tuple[str, ...]
    snapshot: bool

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "parent_version": self.parent_version,
            "content_hash": self.content_hash,
            "message": self.message,
            "author": self.author,
            "tags": list(self.tags),
            "snapshot": self.snapshot,
        }

    @classmethod
    def from_dict(cls, value: dict) -> SuiteVersion:
        return cls(
            version=value["version"],
            created_at=value["created_at"],
            parent_version=value.get("parent_version"),
            content_hash=value["content_hash"],
            message=value.get("message", ""),
            author=value.get("author"),
            tags=tuple(value.get("tags", [])),
            snapshot=value.get("snapshot", False),
        )


@dataclass(frozen=True)
class MigrationPlan:
    """A declarative description of how one suite representation maps onto another."""

    renamed_columns: dict[str, str]
    dropped_columns: frozenset[str]
    type_changes: dict[str, tuple[str, str]]
    scopes: tuple[SuiteBinding, ...]

    def to_dict(self) -> dict:
        return {
            "renamed_columns": dict(self.renamed_columns),
            "dropped_columns": sorted(self.dropped_columns),
            "type_changes": {
                column: {"from": old_type, "to": new_type}
                for column, (old_type, new_type) in self.type_changes.items()
            },
            "scopes": [binding.to_dict() for binding in self.scopes],
        }


@dataclass(frozen=True)
class SuiteBinding:
    """Restricts a migration to expectations associated with one data source and table.

    A binding matches an expectation whose ``meta`` carries the same ``datasource_name`` /
    ``table_name`` (either flat keys or nested under a ``scope`` dict). An empty binding applies
    the migration to every expectation in the suite.
    """

    datasource_name: Optional[str] = None
    table_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {"datasource_name": self.datasource_name, "table_name": self.table_name}

    def matches(self, expectation_dict: dict) -> bool:
        if self.datasource_name is None and self.table_name is None:
            return True
        scope = expectation_dict.get("meta", {}).get("scope", {})
        datasource = expectation_dict.get("meta", {}).get(
            "datasource_name", scope.get("datasource_name")
        )
        table = expectation_dict.get("meta", {}).get("table_name", scope.get("table_name"))
        if self.datasource_name is not None and datasource != self.datasource_name:
            return False
        return self.table_name is None or table == self.table_name


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of applying a :class:`MigrationPlan`."""

    migrated_suite: ExpectationSuite
    affected_expectation_types: tuple[str, ...]
    removed_expectation_types: tuple[str, ...]
    compatibility: CompatibilityReport


# ---------------------------------------------------------------------------
# Canonicalization, hashing and credential redaction
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Stable JSON text for a JSON-serializable value (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def content_hash(value: Any) -> str:
    """Deterministic SHA-256 digest of a JSON-serializable value.

    ``usedforsecurity=False`` keeps the call working on FIPS-enforced interpreters; the hash is
    an integrity fingerprint, not a security primitive.
    """
    digest = hashlib.sha256(canonical_json(value).encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()


def _looks_like_credential_string(value: str) -> bool:
    return bool(_CREDENTIAL_URL_RE.search(value) or _CREDENTIAL_KV_RE.search(value))


def redact_credentials(value: Any) -> Any:
    """Return a copy of ``value`` with anything credential-like replaced.

    Rules:

    * any dict key containing a sensitive marker (``password``, ``token``,
      ``connection_string`` ...) has its value replaced, regardless of nesting depth;
    * any string that embeds a credential URL or a ``password=...`` assignment is replaced.

    The input is never mutated.
    """
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and any(
                marker in key.lower() for marker in SENSITIVE_KEY_MARKERS
            ):
                redacted[key] = REDACTED_PLACEHOLDER
            else:
                redacted[key] = redact_credentials(item)
        return redacted
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_credentials(item) for item in value)
    if isinstance(value, str) and _looks_like_credential_string(value):
        return REDACTED_PLACEHOLDER
    return value


# ---------------------------------------------------------------------------
# JSON Pointer (RFC 6901) helpers and structural diff/patch
# ---------------------------------------------------------------------------


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _parse_pointer(path: str) -> list[Any]:
    if path in ("", "/"):
        return []
    tokens: list[Any] = []
    for raw in path.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if token.lstrip("-").isdigit():
            tokens.append(int(token))
        else:
            tokens.append(token)
    return tokens


def _resolve_parent(value: Any, tokens: Sequence[Any]) -> Any:
    current = value
    for token in list(tokens)[:-1]:
        current = current[token]
    return current


def _list_index(token: Any, length: int) -> int:
    return length if token == "-" else int(token)


def _apply_patch_operation(document: Any, operation: dict) -> Any:
    op = operation["op"]
    tokens = _parse_pointer(operation["path"])
    if not tokens:
        if op != _OP_REPLACE:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Unsupported root-level patch op: {op}"
            )
        return operation["value"]
    parent = _resolve_parent(document, tokens)
    last = tokens[-1]
    if op == _OP_MOVE:
        source_parent = _resolve_parent(document, _parse_pointer(operation["from"]))
        value = source_parent.pop(_parse_pointer(operation["from"])[-1])
    elif op == _OP_ADD:
        value = operation["value"]
    elif op == _OP_REPLACE:
        parent[last] = operation["value"]
        return document
    elif op == _OP_REMOVE:
        parent.pop(_list_index(last, len(parent)) if isinstance(parent, list) else last)
        return document
    else:
        raise SuiteVersionCorruptError(f"Unsupported patch op: {op}")  # noqa: TRY003 # FIXME CoP
    if isinstance(parent, list):
        parent.insert(_list_index(last, len(parent)), value)
    else:
        parent[last] = value
    return document


def apply_patch(document: Any, operations: Sequence[dict]) -> Any:
    """Apply a sequence of JSON-Patch-style operations to a deep-copied document.

    Any failure to resolve a path or apply an op means the stored delta is corrupt; the
    underlying structural error is wrapped in :class:`SuiteVersionCorruptError` so callers
    fail closed rather than seeing a raw ``KeyError``/``TypeError``.
    """
    patched = json.loads(json.dumps(document, default=str))
    for operation in operations:
        try:
            patched = _apply_patch_operation(patched, operation)
        except SuiteVersionCorruptError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Corrupt reverse-delta operation {operation!r}: {exc}"
            ) from exc
    return patched


def _build_reverse_operations(old: Any, new: Any, prefix: str = "") -> list[dict]:
    """Structural diff producing operations that turn ``new`` back into ``old``.

    Lists are compared positionally; an append is recorded as an ``add`` and list-length or
    positional changes as a root ``replace`` of the list.
    """
    operations: list[dict] = []
    if type(old) is not type(new):
        operations.append({"op": _OP_REPLACE, "path": prefix or "/", "value": old})
    elif isinstance(old, dict):
        operations.extend(_reverse_dict_operations(old, new, prefix))
    elif isinstance(old, list):
        operations.extend(_reverse_list_operations(old, new, prefix))
    elif old != new:
        operations.append({"op": _OP_REPLACE, "path": prefix or "/", "value": old})
    return operations


def _reverse_dict_operations(old: dict, new: dict, prefix: str) -> list[dict]:
    operations: list[dict] = []
    for key, old_value in old.items():
        path = f"{prefix}/{_escape_pointer_token(str(key))}"
        if key not in new:
            operations.append({"op": _OP_ADD, "path": path, "value": old_value})
        else:
            operations.extend(_build_reverse_operations(old_value, new[key], path))
    for key in new:
        if key not in old:
            path = f"{prefix}/{_escape_pointer_token(str(key))}"
            operations.append({"op": _OP_REMOVE, "path": path})
    return operations


def _reverse_list_operations(old: list, new: list, prefix: str) -> list[dict]:
    if len(old) != len(new):
        return [{"op": _OP_REPLACE, "path": prefix or "/", "value": old}]
    operations: list[dict] = []
    for index, (old_item, new_item) in enumerate(zip(old, new, strict=True)):
        operations.extend(_build_reverse_operations(old_item, new_item, f"{prefix}/{index}"))
    return operations


# ---------------------------------------------------------------------------
# Suite dictionary helpers
# ---------------------------------------------------------------------------


def _expectation_identity(expectation: dict) -> str:
    """Stable identity for matching the same expectation across two suites.

    Prefer the persisted expectation ``id``; otherwise use expectation type plus domain kwargs
    (column / column_A / column_B / column_list) so a parameter change still matches.
    """
    if expectation.get("id"):
        return f"id:{expectation['id']}"
    kwargs = expectation.get("kwargs", {}) or {}
    domain_parts = [str(expectation.get("type"))]
    for key in ("column", "column_A", "column_B", "column_list"):
        if key in kwargs:
            domain_parts.append(f"{key}={canonical_json(kwargs[key])}")
    return "domain:" + "|".join(domain_parts)


def _suite_dict(suite: ExpectationSuite | dict) -> dict:
    if isinstance(suite, dict):
        return suite
    return suite.to_json_dict()


def _build_suite(suite_dict: dict):
    """Reconstruct an ExpectationSuite, tolerating unregistered custom expectation types."""
    from great_expectations.core.expectation_suite import ExpectationSuite

    try:
        return ExpectationSuite(**suite_dict)
    except ExpectationNotFoundError:
        raise
    except Exception as exc:
        raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
            f"Stored suite payload could not be reconstructed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Field-level diff
# ---------------------------------------------------------------------------

_SUITE_SCALAR_FIELDS = ("name", "notes", "meta", "suite_parameters")


def diff_suites(
    old_suite: ExpectationSuite | dict,
    new_suite: ExpectationSuite | dict,
    version_from: Optional[int] = None,
    version_to: Optional[int] = None,
) -> SuiteDiff:
    """Compute a field-level diff between two suite representations.

    The diff covers individual expectations (added / removed / changed down to the kwarg path),
    suite-level fields (``meta``, ``suite_parameters``, ``notes`` ...) and a per-column summary
    that classifies column additions, removals, renames and type changes.
    """
    old = _suite_dict(old_suite)
    new = _suite_dict(new_suite)

    old_expectations = {_expectation_identity(item): item for item in old.get("expectations", [])}
    new_expectations = {_expectation_identity(item): item for item in new.get("expectations", [])}

    expectation_changes: list[FieldChange] = []
    for identity in sorted(new_expectations.keys() - old_expectations.keys()):
        expectation_changes.append(
            FieldChange(
                path=f"/expectations/{identity}",
                op=_OP_ADD,
                old=None,
                new=new_expectations[identity],
            )
        )
    for identity in sorted(old_expectations.keys() - new_expectations.keys()):
        expectation_changes.append(
            FieldChange(
                path=f"/expectations/{identity}",
                op=_OP_REMOVE,
                old=old_expectations[identity],
                new=None,
            )
        )
    for identity in sorted(old_expectations.keys() & new_expectations.keys()):
        old_item = old_expectations[identity]
        new_item = new_expectations[identity]
        if canonical_json(old_item) != canonical_json(new_item):
            for operation in _build_reverse_operations(old_item, new_item, ""):
                # Reverse operations describe old->new from the inverse direction; translate to
                # forward field changes with both values attached.
                path = f"/expectations/{identity}{operation['path']}".rstrip("/") or (
                    f"/expectations/{identity}"
                )
                old_value, new_value = _values_at_path(old_item, new_item, operation["path"])
                expectation_changes.append(
                    FieldChange(
                        path=path, op=_forward_op(operation["op"]), old=old_value, new=new_value
                    )
                )

    field_changes: list[FieldChange] = []
    for name in _SUITE_SCALAR_FIELDS:
        if canonical_json(old.get(name)) != canonical_json(new.get(name)):
            field_changes.append(
                FieldChange(path=f"/{name}", op=_OP_REPLACE, old=old.get(name), new=new.get(name))
            )

    column_changes = _diff_columns(old_expectations, new_expectations, expectation_changes)
    has_changes = bool(expectation_changes or field_changes)
    return SuiteDiff(
        version_from=version_from,
        version_to=version_to,
        expectation_changes=tuple(expectation_changes),
        column_changes=tuple(column_changes),
        field_changes=tuple(field_changes),
        has_changes=has_changes,
    )


def _forward_op(reverse_op: str) -> str:
    if reverse_op == _OP_ADD:
        return _OP_REMOVE
    if reverse_op == _OP_REMOVE:
        return _OP_ADD
    return reverse_op


def _values_at_path(old_item: dict, new_item: dict, path: str) -> tuple[Any, Any]:
    tokens = _parse_pointer(path)
    old_value: Any = old_item
    new_value: Any = new_item
    for token in tokens[:-1]:
        old_value = old_value[token] if isinstance(old_value, dict) and token in old_value else None
        new_value = new_value[token] if isinstance(new_value, dict) and token in new_value else None
    if not tokens:
        return old_item, new_item
    token = tokens[-1]
    old_leaf = old_value.get(token) if isinstance(old_value, dict) else None
    new_leaf = new_value.get(token) if isinstance(new_value, dict) else None
    return old_leaf, new_leaf


def _columns_in(expectations: dict[str, dict]) -> dict[str, set[str]]:
    columns: dict[str, set[str]] = {}
    for expectation in expectations.values():
        column = (expectation.get("kwargs") or {}).get("column")
        if column:
            columns.setdefault(str(column), set()).add(str(expectation.get("type")))
    return columns


def _detect_column_renames(
    old_columns: dict[str, set[str]],
    new_columns: dict[str, set[str]],
) -> dict[str, str]:
    """Map old column names to new ones for columns that were renamed.

    A column that exists on only one side is treated as a rename when the same set of
    expectation types applies to it on both sides (e.g. ``expect_column_values_to_not_be_null``
    moved from ``old_name`` to ``new_name``), rather than reporting an unrelated add/remove
    pair. The match must be 1:1: if several old or new columns share the same type set, the
    situation is ambiguous and those columns are left as add/remove rather than guessed.
    """
    renamed: dict[str, str] = {}
    only_old = sorted(set(old_columns) - set(new_columns))
    only_new = sorted(set(new_columns) - set(old_columns))
    for removed in only_old:
        candidates = [added for added in only_new if new_columns[added] == old_columns[removed]]
        same_type_old = [
            column for column in only_old if old_columns[column] == old_columns[removed]
        ]
        if (
            len(candidates) == 1
            and len(same_type_old) == 1
            and candidates[0] not in renamed.values()
        ):
            renamed[removed] = candidates[0]
    return renamed


def _column_type_changes(
    common_columns: set[str],
    old_expectations: dict[str, dict],
    new_expectations: dict[str, dict],
) -> list[ColumnChange]:
    changes: list[ColumnChange] = []
    for column in sorted(common_columns):
        type_change = _column_type_change(column, old_expectations, new_expectations)
        if not type_change:
            continue
        old_type, new_type = type_change
        changes.append(
            ColumnChange(
                column=column,
                change_type="type_changed",
                details=(
                    FieldChange(
                        path=f"/column/{_escape_pointer_token(column)}/type",
                        op=_OP_REPLACE,
                        old=old_type,
                        new=new_type,
                    ),
                ),
            )
        )
    return changes


def _diff_columns(
    old_expectations: dict[str, dict],
    new_expectations: dict[str, dict],
    expectation_changes: Sequence[FieldChange],
) -> list[ColumnChange]:
    old_columns = _columns_in(old_expectations)
    new_columns = _columns_in(new_expectations)

    # A column present in only one side whose expectations also appear (renamed) on the other
    # side is reported as a rename rather than an add/remove pair.
    renamed = _detect_column_renames(old_columns, new_columns)

    changes: list[ColumnChange] = []
    for old_name, new_name in sorted(renamed.items()):
        changes.append(
            ColumnChange(
                column=f"{old_name} -> {new_name}",
                change_type="renamed",
                details=tuple(
                    change
                    for change in expectation_changes
                    if _escape_pointer_token(old_name) in change.path
                    or _escape_pointer_token(new_name) in change.path
                ),
            )
        )
    for column in sorted(set(old_columns) - set(new_columns) - set(renamed)):
        changes.append(ColumnChange(column=column, change_type="removed"))
    for column in sorted(set(new_columns) - set(old_columns) - set(renamed.values())):
        changes.append(ColumnChange(column=column, change_type="added"))

    common_columns = set(old_columns) & set(new_columns)
    changes.extend(_column_type_changes(common_columns, old_expectations, new_expectations))
    return changes


def _column_type_change(
    column: str, old_expectations: dict[str, dict], new_expectations: dict[str, dict]
) -> Optional[tuple[str, str]]:
    def _type_token(expectations: dict[str, dict]) -> Optional[str]:
        for expectation in expectations.values():
            kwargs = expectation.get("kwargs") or {}
            if kwargs.get("column") != column:
                continue
            # ``type`` is the expectation-type key and must never be confused with a column
            # data type; only explicit data-type kwargs are considered.
            for key in ("type_", "dtype", "column_type"):
                if key in kwargs:
                    return str(kwargs[key])
        return None

    old_type = _type_token(old_expectations)
    new_type = _type_token(new_expectations)
    if old_type and new_type and old_type != new_type:
        return old_type, new_type
    return None


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------


def _iter_parameter_refs(value: Any) -> Any:
    if isinstance(value, dict):
        if "$PARAMETER" in value:
            yield str(value["$PARAMETER"])
        for item in value.values():
            yield from _iter_parameter_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_parameter_refs(item)
    elif isinstance(value, str) and value.startswith("$") and len(value) > 1:
        yield value[1:]


def _expectation_change_issues(diff: SuiteDiff) -> list[CompatibilityIssue]:
    from great_expectations.expectations.registry import (
        _registered_expectations,
    )

    issues: list[CompatibilityIssue] = []
    for change in diff.expectation_changes:
        if change.op == _OP_REMOVE:
            expectation_type = (change.old or {}).get("type", "unknown")
            issues.append(
                CompatibilityIssue(
                    severity="breaking",
                    code="EXPECTATION_REMOVED",
                    message=f"Expectation {expectation_type} was removed.",
                    path=change.path,
                )
            )
        elif change.op == _OP_ADD:
            expectation_type = (change.new or {}).get("type", "unknown")
            if expectation_type not in _registered_expectations:
                issues.append(
                    CompatibilityIssue(
                        severity="warning",
                        code="CUSTOM_EXPECTATION_UNREGISTERED",
                        message=(
                            f"Expectation type {expectation_type} is not registered in this "
                            "process; it may be a custom expectation unavailable here."
                        ),
                        path=change.path,
                    )
                )
    return issues


def _column_change_issues(diff: SuiteDiff) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    for column_change in diff.column_changes:
        if column_change.change_type == "removed":
            issues.append(
                CompatibilityIssue(
                    severity="breaking",
                    code="COLUMN_REMOVED",
                    message=(
                        f"Column {column_change.column} is referenced by the old suite but no "
                        "longer exists."
                    ),
                    path=f"/column/{_escape_pointer_token(column_change.column)}",
                )
            )
        elif column_change.change_type == "type_changed":
            details = column_change.details[0]
            issues.append(
                CompatibilityIssue(
                    severity="breaking",
                    code="COLUMN_TYPE_CHANGED",
                    message=(
                        f"Column {column_change.column} type changed from {details.old} to "
                        f"{details.new}; expectations assuming the old type may fail."
                    ),
                    path=details.path,
                )
            )
    return issues


def _suite_parameter_issues(old: dict, new: dict) -> list[CompatibilityIssue]:
    old_parameters = set((old.get("suite_parameters") or {}).keys())
    new_parameters = set((new.get("suite_parameters") or {}).keys())
    referenced = {ref for item in new.get("expectations", []) for ref in _iter_parameter_refs(item)}
    issues: list[CompatibilityIssue] = []
    for missing in sorted((old_parameters - new_parameters) & referenced):
        issues.append(
            CompatibilityIssue(
                severity="breaking",
                code="SUITE_PARAMETER_REMOVED",
                message=f"Suite parameter ${missing} is still referenced but was removed.",
                path=f"/suite_parameters/{_escape_pointer_token(missing)}",
            )
        )
    return issues


def _expected_type_issues(
    new: dict, expected_column_types: dict[str, str]
) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    for expectation in new.get("expectations", []):
        kwargs = expectation.get("kwargs") or {}
        column = kwargs.get("column")
        if column not in expected_column_types:
            continue
        for key in ("type_", "dtype", "column_type"):
            if key in kwargs and str(kwargs[key]) != expected_column_types[column]:
                issues.append(
                    CompatibilityIssue(
                        severity="breaking",
                        code="COLUMN_TYPE_MISMATCH",
                        message=(
                            f"Expectation {expectation.get('type')} declares {column} as "
                            f"{kwargs[key]} but the target type is "
                            f"{expected_column_types[column]}."
                        ),
                        path=f"/expectations/{_expectation_identity(expectation)}/kwargs/{key}",
                    )
                )
    return issues


def check_suite_compatibility(
    old_suite: ExpectationSuite | dict,
    new_suite: ExpectationSuite | dict,
    *,
    expected_column_types: Optional[dict[str, str]] = None,
) -> CompatibilityReport:
    """Check whether ``new_suite`` is a safe evolution of ``old_suite``.

    ``expected_column_types`` maps a column name to the type the migrated data will actually
    have; expectations whose declared type disagrees with it are reported as breaking.

    A report is *compatible* only when it contains no ``breaking`` issues. ``warning`` issues
    (for example custom expectation types that are not registered in this process) do not block
    a migration but are surfaced for review.
    """
    old = _suite_dict(old_suite)
    new = _suite_dict(new_suite)
    diff = diff_suites(old, new)
    issues: list[CompatibilityIssue] = []
    issues.extend(_expectation_change_issues(diff))
    issues.extend(_column_change_issues(diff))
    issues.extend(_suite_parameter_issues(old, new))
    if expected_column_types:
        issues.extend(_expected_type_issues(new, expected_column_types))

    compatible = not any(issue.severity == "breaking" for issue in issues)
    return CompatibilityReport(compatible=compatible, issues=tuple(issues))


# ---------------------------------------------------------------------------
# Migration planning and application
# ---------------------------------------------------------------------------


def build_migration_plan(
    old_suite: ExpectationSuite | dict,
    new_suite: ExpectationSuite | dict,
    *,
    scopes: Optional[Sequence[SuiteBinding]] = None,
) -> MigrationPlan:
    """Infer a :class:`MigrationPlan` from the diff of two suite representations."""
    diff = diff_suites(old_suite, new_suite)
    renamed_columns: dict[str, str] = {}
    dropped_columns: set[str] = set()
    type_changes: dict[str, tuple[str, str]] = {}
    for column_change in diff.column_changes:
        if column_change.change_type == "renamed":
            old_name, new_name = column_change.column.split(" -> ", 1)
            renamed_columns[old_name] = new_name
        elif column_change.change_type == "removed":
            dropped_columns.add(column_change.column)
        elif column_change.change_type == "type_changed" and column_change.details:
            detail = column_change.details[0]
            type_changes[column_change.column] = (str(detail.old), str(detail.new))
    return MigrationPlan(
        renamed_columns=renamed_columns,
        dropped_columns=frozenset(dropped_columns),
        type_changes=type_changes,
        scopes=tuple(scopes or (SuiteBinding(),)),
    )


def _plan_matches(plan: MigrationPlan, expectation_dict: dict) -> bool:
    # A plan without explicit scopes applies to every expectation in the suite.
    if not plan.scopes:
        return True
    return any(binding.matches(expectation_dict) for binding in plan.scopes)


_TYPE_KWARG_KEYS = ("type_", "dtype", "column_type")


def _migrate_column_pairs_and_lists(
    kwargs: dict, plan: MigrationPlan, expectation_type: str
) -> set[str]:
    affected: set[str] = set()
    for pair_key in ("column_A", "column_B"):
        if kwargs.get(pair_key) in plan.renamed_columns:
            kwargs[pair_key] = plan.renamed_columns[kwargs[pair_key]]
            affected.add(expectation_type)
    if isinstance(kwargs.get("column_list"), list):
        kwargs["column_list"] = [
            plan.renamed_columns.get(item, item)
            for item in kwargs["column_list"]
            if item not in plan.dropped_columns
        ]
        affected.add(expectation_type)
    return affected


def _migrate_column_type(
    kwargs: dict, plan: MigrationPlan, column: Optional[str], expectation_type: str
) -> set[str]:
    if column not in plan.type_changes:
        return set()
    affected: set[str] = set()
    for key in _TYPE_KWARG_KEYS:
        if key in kwargs:
            kwargs[key] = plan.type_changes[column][1]
            affected.add(expectation_type)
    return affected


def _migrate_expectation(
    expectation: dict, plan: MigrationPlan
) -> tuple[Optional[dict], Optional[str], set[str]]:
    """Apply a plan to one expectation dict.

    Returns ``(kept_expectation, removed_type, affected_types)``; when the column is dropped,
    ``kept_expectation`` is None and ``removed_type`` is set.
    """
    kwargs = expectation.get("kwargs") or {}
    column = kwargs.get("column")
    expectation_type = str(expectation.get("type"))
    if column in plan.dropped_columns:
        return None, expectation_type, set()
    affected: set[str] = set()
    if column in plan.renamed_columns:
        kwargs["column"] = plan.renamed_columns[column]
        affected.add(expectation_type)
    affected.update(_migrate_column_pairs_and_lists(kwargs, plan, expectation_type))
    affected.update(_migrate_column_type(kwargs, plan, column, expectation_type))
    return expectation, None, affected


def _apply_plan_to_dict(suite_dict: dict, plan: MigrationPlan) -> tuple[dict, list[str], list[str]]:
    migrated = json.loads(json.dumps(suite_dict, default=str))
    kept: list[dict] = []
    affected: set[str] = set()
    removed: set[str] = set()
    for expectation in migrated.get("expectations", []):
        if not _plan_matches(plan, expectation):
            kept.append(expectation)
            continue
        migrated_expectation, removed_type, affected_types = _migrate_expectation(expectation, plan)
        affected.update(affected_types)
        if removed_type is not None:
            removed.add(removed_type)
            continue
        kept.append(migrated_expectation)
    migrated["expectations"] = kept
    return migrated, sorted(affected), sorted(removed)


def apply_migration(
    suite: ExpectationSuite,
    plan: MigrationPlan,
    *,
    expected_column_types: Optional[dict[str, str]] = None,
    strict: bool = True,
) -> MigrationResult:
    """Apply a :class:`MigrationPlan` to a suite and verify compatibility.

    When ``strict`` is True (default), a breaking :class:`CompatibilityReport` raises
    :class:`SuiteMigrationError` and no suite is returned, so callers cannot half-migrate.
    """
    suite_dict = suite.to_json_dict()
    migrated_dict, affected, removed = _apply_plan_to_dict(suite_dict, plan)
    report = check_suite_compatibility(
        suite_dict, migrated_dict, expected_column_types=expected_column_types
    )
    if strict and not report.compatible:
        messages = "; ".join(
            issue.message for issue in report.issues if issue.severity == "breaking"
        )
        raise SuiteMigrationError(f"Refusing to apply incompatible migration: {messages}")  # noqa: TRY003 # FIXME CoP
    migrated_suite = _build_suite(migrated_dict)
    return MigrationResult(
        migrated_suite=migrated_suite,
        affected_expectation_types=tuple(affected),
        removed_expectation_types=tuple(removed),
        compatibility=report,
    )


# ---------------------------------------------------------------------------
# Cross-process file lock
# ---------------------------------------------------------------------------


class _FileLock:
    """Best-effort advisory lock.

    Uses ``fcntl.flock`` where available (POSIX). On platforms without ``fcntl`` the lock
    degrades to in-process serialization so the API stays uniform; the on-disk format itself
    remains portable.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Optional[Any] = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a+")
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX platforms
            logger.debug("fcntl unavailable; SuiteVersionStore lock is in-process only")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover
            pass
        self._handle.close()
        self._handle = None


# ---------------------------------------------------------------------------
# Immutable, delta-based version store
# ---------------------------------------------------------------------------


class SuiteVersionStore:
    """File-backed, immutable version store for one or more Expectation Suites.

    Layout under ``base_directory``::

        <base_directory>/<suite slug>/
            versions.jsonl          # append-only reverse-delta log + version metadata (bounded)
            snapshots/<shard>/v<N>.json  # bounded full-suite snapshot cache
            .lock                   # cross-process advisory lock

    The store never duplicates the whole suite per version. Each commit writes a snapshot of
    the current head and a reverse delta against its parent; an older version is reconstructed
    by reverse-applying deltas from the nearest retained later snapshot. Both the delta log
    (``max_versions``) and the snapshot cache (``snapshot_cache_size``) have hard upper bounds,
    so total on-disk size is bounded by::

        max_versions * (delta size) + snapshot_cache_size * (full suite size)
    """

    def __init__(
        self,
        base_directory: str | Path,
        *,
        max_versions: int = DEFAULT_MAX_VERSIONS,
        snapshot_cache_size: int = DEFAULT_SNAPSHOT_CACHE_SIZE,
    ) -> None:
        if max_versions < 1:
            raise ValueError("max_versions must be >= 1")  # noqa: TRY003 # FIXME CoP
        # At least two snapshots are required for the reconstruction invariant: the oldest
        # retained version needs an anchor with no earlier state, and every other version needs
        # a strictly later snapshot to reverse-apply deltas from.
        if snapshot_cache_size < MIN_SNAPSHOT_CACHE_SIZE:
            raise ValueError(  # noqa: TRY003 # FIXME CoP
                f"snapshot_cache_size must be >= {MIN_SNAPSHOT_CACHE_SIZE}"
            )
        self.base_directory = Path(base_directory)
        self.max_versions = max_versions
        self.snapshot_cache_size = snapshot_cache_size

    # -- path helpers ------------------------------------------------------

    @staticmethod
    def _slugify(suite_name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", suite_name).strip("._")
        return slug or "suite"

    def _suite_dir(self, suite_name: str) -> Path:
        return self.base_directory / self._slugify(suite_name)

    def _log_path(self, suite_name: str) -> Path:
        return self._suite_dir(suite_name) / _LOG_FILE_NAME

    def _lock(self, suite_name: str) -> _FileLock:
        return _FileLock(self._suite_dir(suite_name) / _LOCK_FILE_NAME)

    def _snapshot_path(self, suite_name: str, version: int) -> Path:
        shard = version // _SHARD_SIZE
        return (
            self._suite_dir(suite_name)
            / _SNAPSHOTS_DIR_NAME
            / f"shard_{shard:04d}"
            / f"v{version}.json"
        )

    # -- low level record IO ----------------------------------------------

    def _read_records(self, suite_name: str) -> list[dict]:
        log_path = self._log_path(suite_name)
        if not log_path.exists():
            return []
        records: list[dict] = []
        with open(log_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                        f"Corrupt version log line {line_number} for suite {suite_name!r}: {exc}"
                    ) from exc
                self._validate_record(record, suite_name, line_number)
                records.append(record)
        return records

    @staticmethod
    def _validate_record(record: Any, suite_name: str, line_number: int) -> None:
        if not isinstance(record, dict) or "version" not in record or "content_hash" not in record:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Malformed version record on line {line_number} for suite {suite_name!r}"
            )
        if not record.get("snapshot") and "reverse_patch" not in record:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Delta record on line {line_number} for suite {suite_name!r} is missing "
                "its reverse_patch"
            )

    def _append_record(self, suite_name: str, record: dict) -> None:
        log_path = self._log_path(suite_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # All writers hold the suite lock, and O_APPEND guarantees the single line is written
        # atomically, so concurrent processes can never interleave bytes inside a record.
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str, ensure_ascii=False) + "\n")

    def _write_snapshot(self, suite_name: str, version: int, payload: dict) -> None:
        snapshot_path = self._snapshot_path(suite_name, version)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = snapshot_path.with_suffix(".tmp")
        envelope = {
            "version": version,
            "content_hash": content_hash(payload),
            "suite": redact_credentials(payload),
        }
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, sort_keys=True, default=str, ensure_ascii=False)
        Path(tmp_path).replace(snapshot_path)

    def _read_snapshot(self, suite_name: str, version: int) -> dict:
        snapshot_path = self._snapshot_path(suite_name, version)
        if not snapshot_path.exists():
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Snapshot v{version} for suite {suite_name!r} is missing"
            )
        try:
            with open(snapshot_path, encoding="utf-8") as handle:
                envelope = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Corrupt snapshot v{version} for suite {suite_name!r}: {exc}"
            ) from exc
        payload = envelope.get("suite")
        if not isinstance(payload, dict) or envelope.get("content_hash") != content_hash(payload):
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Snapshot v{version} for suite {suite_name!r} failed its integrity check"
            )
        return payload

    # -- version reconstruction -------------------------------------------

    def _payload_at(self, suite_name: str, records: list[dict], version: int) -> dict:
        target = next((record for record in records if record["version"] == version), None)
        if target is None:
            raise SuiteVersionNotFoundError(f"Version {version} not found for suite {suite_name!r}")  # noqa: TRY003 # FIXME CoP
        if target.get("snapshot"):
            return self._read_snapshot(suite_name, version)
        # Every commit is written with a snapshot of the current head. A non-snapshot version
        # therefore always has at least one later snapshot; walk to the nearest one and
        # reverse-apply the deltas back to the target. Reverse patches carry the old value on
        # every op, so no information is needed from the pre-image being reconstructed.
        later_snapshots = [
            record["version"]
            for record in records
            if record["version"] > version and record.get("snapshot")
        ]
        if not later_snapshots:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"No later snapshot found to reconstruct v{version} of {suite_name!r}"
            )
        anchor = min(later_snapshots)
        payload = self._read_snapshot(suite_name, anchor)
        for record in sorted(
            (item for item in records if version < item["version"] <= anchor),
            key=lambda item: item["version"],
            reverse=True,
        ):
            payload = apply_patch(payload, record["reverse_patch"])
        if content_hash(payload) != target["content_hash"]:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Reconstructed v{version} of {suite_name!r} failed its integrity check"
            )
        return payload

    # -- public API --------------------------------------------------------

    def commit(
        self,
        suite: ExpectationSuite,
        *,
        message: str = "",
        author: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        expected_parent_version: Optional[int] = None,
    ) -> SuiteVersion:
        """Persist a new immutable version of ``suite``.

        The whole operation runs under the suite lock and validates the payload before anything
        is written, so a failure leaves every previously committed version untouched.

        Raises:
            SuiteVersionConflictError: ``expected_parent_version`` does not match the current
                head (optimistic concurrency control for concurrent editors).
        """
        suite_name = suite.name
        with self._lock(suite_name):
            records = self._read_records(suite_name)
            head = records[-1]["version"] if records else None
            if expected_parent_version is not None and expected_parent_version != head:
                raise SuiteVersionConflictError(  # noqa: TRY003 # FIXME CoP
                    f"Expected parent v{expected_parent_version} but head is "
                    f"v{head if head is not None else '<none>'} for suite {suite_name!r}"
                )

            payload = redact_credentials(suite.to_json_dict())
            # Fail fast on a payload that cannot be round-tripped before touching the log.
            self._assert_reconstructable(payload)
            new_version = (head or 0) + 1
            # The head is always snapshotted so any retained version can be rebuilt by
            # reverse-applying deltas from the nearest later snapshot. Older snapshots are
            # evicted afterwards by the bounded snapshot cache.
            record: dict[str, Any] = {
                "version": new_version,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "parent_version": head,
                "content_hash": content_hash(payload),
                "message": redact_credentials(message),
                "author": redact_credentials(author),
                "tags": list(redact_credentials(list(tags or []))),
                "snapshot": True,
            }
            if head is not None:
                parent_payload = self._payload_at(suite_name, records, head)
                record["reverse_patch"] = _build_reverse_operations(parent_payload, payload)
            self._write_snapshot(suite_name, new_version, payload)
            self._append_record(suite_name, record)
            self._prune(suite_name)
            return SuiteVersion.from_dict(record)

    @staticmethod
    def _assert_reconstructable(payload: dict) -> None:
        try:
            json.loads(json.dumps(payload, default=str))
        except (TypeError, ValueError) as exc:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Suite payload is not JSON-serializable: {exc}"
            ) from exc

    def list_versions(self, suite_name: str) -> list[SuiteVersion]:
        with self._lock(suite_name):
            return [SuiteVersion.from_dict(record) for record in self._read_records(suite_name)]

    def head_version(self, suite_name: str) -> Optional[SuiteVersion]:
        versions = self.list_versions(suite_name)
        return versions[-1] if versions else None

    def load_suite_dict(self, suite_name: str, version: Optional[int] = None) -> dict:
        with self._lock(suite_name):
            records = self._read_records(suite_name)
            if not records:
                raise SuiteVersionNotFoundError(f"No versions exist for suite {suite_name!r}")  # noqa: TRY003 # FIXME CoP
            target = version if version is not None else records[-1]["version"]
            return self._payload_at(suite_name, records, target)

    def load_suite(self, suite_name: str, version: Optional[int] = None) -> ExpectationSuite:
        return _build_suite(self.load_suite_dict(suite_name, version))

    def diff(
        self,
        suite_name: str,
        version_from: Optional[int] = None,
        version_to: Optional[int] = None,
    ) -> SuiteDiff:
        """Field-level diff between two committed versions (defaults: v1 -> head)."""
        with self._lock(suite_name):
            records = self._read_records(suite_name)
            if not records:
                raise SuiteVersionNotFoundError(f"No versions exist for suite {suite_name!r}")  # noqa: TRY003 # FIXME CoP
            low = version_from if version_from is not None else records[0]["version"]
            high = version_to if version_to is not None else records[-1]["version"]
            if low > high:
                raise SuiteVersionError("version_from must be <= version_to")  # noqa: TRY003 # FIXME CoP
            old_payload = self._payload_at(suite_name, records, low)
            new_payload = self._payload_at(suite_name, records, high)
            return diff_suites(old_payload, new_payload, low, high)

    def rollback(
        self,
        suite_name: str,
        version: int,
        *,
        message: str = "",
        author: Optional[str] = None,
    ) -> SuiteVersion:
        """Restore the content of an older version as a *new* immutable version.

        History is never rewritten: the rollback target stays reachable and the new head is a
        fresh commit whose payload equals v``version``, making the rollback itself auditable.
        """
        with self._lock(suite_name):
            records = self._read_records(suite_name)
            if not records:
                raise SuiteVersionNotFoundError(f"No versions exist for suite {suite_name!r}")  # noqa: TRY003 # FIXME CoP
            target_payload = self._payload_at(suite_name, records, version)
            head = records[-1]["version"]
            new_version = head + 1
            record: dict[str, Any] = {
                "version": new_version,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "parent_version": head,
                "content_hash": content_hash(target_payload),
                "message": message or f"Rollback to v{version}",
                "author": redact_credentials(author),
                "tags": ["rollback"],
                "snapshot": True,
            }
            parent_payload = self._payload_at(suite_name, records, head)
            record["reverse_patch"] = _build_reverse_operations(parent_payload, target_payload)
            self._write_snapshot(suite_name, new_version, target_payload)
            self._append_record(suite_name, record)
            self._prune(suite_name)
            return SuiteVersion.from_dict(record)

    def repair(self, suite_name: str, *, dry_run: bool = True) -> list[str]:
        """List and, with ``dry_run=False``, rebuild corrupt snapshot cache files.

        The delta log is the source of truth and is never rewritten. A corrupt snapshot is
        rebuilt from the reverse-delta chain of a later intact snapshot, so every retained
        version stays reachable. If the corrupt snapshot is the only remaining anchor (no
        later snapshot exists), the error is reported and nothing is deleted.
        """
        corrupt: list[str] = []
        with self._lock(suite_name):
            records = self._read_records(suite_name)
            later_anchor_versions = [
                record["version"] for record in records if record.get("snapshot")
            ]
            for record in records:
                if not record.get("snapshot"):
                    continue
                version = record["version"]
                snapshot_path = self._snapshot_path(suite_name, version)
                try:
                    self._read_snapshot(suite_name, version)
                except SuiteVersionCorruptError:
                    corrupt.append(str(snapshot_path))
                    if dry_run:
                        continue
                    anchors = [
                        candidate for candidate in later_anchor_versions if candidate > version
                    ]
                    if not anchors:
                        raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                            f"Cannot rebuild v{version} of {suite_name!r}: no later intact "
                            "snapshot exists to reconstruct it from"
                        )
                    # Rebuild via the nearest later anchor and verify the recorded hash.
                    payload = self._rebuild_from_later_snapshot(suite_name, records, version)
                    self._write_snapshot(suite_name, version, payload)
        return corrupt

    def _rebuild_from_later_snapshot(
        self, suite_name: str, records: list[dict], version: int
    ) -> dict:
        later_snapshots = [
            record["version"]
            for record in records
            if record["version"] > version and record.get("snapshot")
        ]
        anchor = min(later_snapshots)
        payload = self._read_snapshot(suite_name, anchor)
        for record in sorted(
            (item for item in records if version < item["version"] <= anchor),
            key=lambda item: item["version"],
            reverse=True,
        ):
            payload = apply_patch(payload, record["reverse_patch"])
        target = next(item for item in records if item["version"] == version)
        if content_hash(payload) != target["content_hash"]:
            raise SuiteVersionCorruptError(  # noqa: TRY003 # FIXME CoP
                f"Rebuilt v{version} of {suite_name!r} failed its integrity check"
            )
        return payload

    # -- retention ---------------------------------------------------------

    def _prune(self, suite_name: str) -> None:
        """Enforce the bounded log and snapshot cache.

        Oldest delta records are dropped first. If pruning would remove the only anchor a
        surviving version needs, a fresh snapshot of the new oldest version is written first,
        so every retained version stays reconstructable.
        """
        records = self._read_records(suite_name)
        if len(records) <= self.max_versions:
            self._prune_snapshots(suite_name, records)
            return
        keep = records[-self.max_versions :]
        oldest_kept = keep[0]
        if not oldest_kept.get("snapshot"):
            payload = self._payload_at(suite_name, records, oldest_kept["version"])
            oldest_kept = dict(oldest_kept)
            oldest_kept["snapshot"] = True
            oldest_kept.pop("reverse_patch", None)
            self._write_snapshot(suite_name, oldest_kept["version"], payload)
            keep[0] = oldest_kept
        removed_versions = {record["version"] for record in records[: -self.max_versions]}
        self._rewrite_log(suite_name, keep)
        for version in removed_versions:
            self._snapshot_path(suite_name, version).unlink(missing_ok=True)
        self._prune_snapshots(suite_name, keep)

    def _ensure_oldest_anchor(self, suite_name: str, records: list[dict]) -> bool:
        """Guarantee the oldest retained version has a snapshot anchor.

        Returns True when the log was rewritten as a result.
        """
        anchor_record = records[0]
        if anchor_record.get("snapshot"):
            return False
        payload = self._rebuild_from_later_snapshot(suite_name, records, anchor_record["version"])
        self._write_snapshot(suite_name, anchor_record["version"], payload)
        anchor_record["snapshot"] = True
        anchor_record.pop("reverse_patch", None)
        self._rewrite_log(suite_name, records)
        return True

    def _select_retained_snapshot_versions(
        self, candidates: list[dict], keep_count: int
    ) -> set[int]:
        """Pick an evenly spaced sample that always includes the newest candidate."""
        if keep_count == 1:
            return {candidates[-1]["version"]}
        step = (len(candidates) - 1) / (keep_count - 1)
        return {candidates[round(index * step)]["version"] for index in range(keep_count)}

    def _prune_snapshots(self, suite_name: str, records: list[dict]) -> None:
        """Bound the snapshot cache while keeping every retained version reconstructable.

        The oldest retained version is always anchored (it has no earlier state to rebuild
        from). If its snapshot was evicted earlier it is rebuilt from the next snapshot before
        evicting anything else. Among the remaining snapshots we keep a bounded, evenly spaced
        sample, so the maximum reverse-replay distance stays proportional to
        ``max_versions / snapshot_cache_size``.
        """
        if not records:
            return
        anchored_version = records[0]["version"]
        self._ensure_oldest_anchor(suite_name, records)

        snapshot_records = [record for record in records if record.get("snapshot")]
        if len(snapshot_records) <= self.snapshot_cache_size:
            return
        candidates = [
            record for record in snapshot_records if record["version"] != anchored_version
        ]
        keep_count = self.snapshot_cache_size - 1
        if len(candidates) <= keep_count:
            return
        chosen = self._select_retained_snapshot_versions(candidates, keep_count)
        evicted = False
        for record in candidates:
            if record["version"] in chosen:
                continue
            self._snapshot_path(suite_name, record["version"]).unlink(missing_ok=True)
            record["snapshot"] = False
            evicted = True
        if evicted:
            # The log is the source of truth; keep its snapshot flags in sync with the cache.
            self._rewrite_log(suite_name, records)

    def _rewrite_log(self, suite_name: str, records: list[dict]) -> None:
        log_path = self._log_path(suite_name)
        tmp_path = log_path.with_name(log_path.name + ".rewrite")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, sort_keys=True, default=str, ensure_ascii=False) + "\n"
                )
        Path(tmp_path).replace(log_path)


# ---------------------------------------------------------------------------
# Convenience facade: migration + versioning together, scope-aware
# ---------------------------------------------------------------------------


class SuiteMigrator:
    """Apply migrations to suites and record each migration as an immutable version."""

    def __init__(self, store: SuiteVersionStore) -> None:
        self.store = store

    def migrate(  # noqa: PLR0913 # FIXME CoP
        self,
        suite: ExpectationSuite,
        plan: MigrationPlan,
        *,
        expected_parent_version: Optional[int] = None,
        expected_column_types: Optional[dict[str, str]] = None,
        strict: bool = True,
        author: Optional[str] = None,
    ) -> tuple[MigrationResult, SuiteVersion]:
        """Migrate ``suite`` under ``plan`` and commit the result as a new version.

        The compatibility check runs before anything is written; a breaking report raises in
        strict mode and leaves the store (and the input suite) untouched.
        """
        result = apply_migration(
            suite, plan, expected_column_types=expected_column_types, strict=strict
        )
        version = self.store.commit(
            result.migrated_suite,
            message=f"Migration: {plan.to_dict()}",
            author=author,
            tags=["migration"],
            expected_parent_version=expected_parent_version,
        )
        return result, version

    def rollback(
        self,
        suite_name: str,
        version: int,
        *,
        author: Optional[str] = None,
    ) -> ExpectationSuite:
        self.store.rollback(suite_name, version, author=author)
        return self.store.load_suite(suite_name)
