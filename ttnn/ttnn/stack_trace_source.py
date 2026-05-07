# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for resolving Python stack traces to on-disk source and SQLite source_files rows."""

import functools
import os
import re
import site
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path

_PYTHON_STACK_FILE_PATTERN = re.compile(r'^\s*File "([^"]+)", line \d+, in ')
_PATH_WITH_LINE_PATTERN = re.compile(r"^\s*([^:\n]+):(\d+)(?::\d+)?\s*$")

# Shared by ``graph_report.create_database_schema`` and ``database.get_or_create_sqlite_db``.
CREATE_SOURCE_FILES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY,
    path text UNIQUE NOT NULL,
    contents text
)
"""

CREATE_STACK_TRACES_TABLE_WITH_SOURCE_SQL = """
CREATE TABLE IF NOT EXISTS stack_traces (
    operation_id int,
    stack_trace text,
    source_file_id int REFERENCES source_files(id)
)
"""

CREATE_INDEX_STACK_TRACES_SOURCE_FILE_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_stack_traces_source_file_id ON stack_traces (source_file_id)"
)


def _realpath(path: str) -> str | None:
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return None


def _is_regular_file(path: str) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode)


def _path_is_under_root(candidate: str, root: str) -> bool:
    if candidate == root:
        return True
    prefix = root if root.endswith(os.sep) else root + os.sep
    return candidate.startswith(prefix)


@functools.lru_cache(maxsize=64)
def _allowed_stack_trace_source_roots(
    cwd: str,
    sys_prefix: str,
    sys_base_prefix: str,
    venv: str | None,
    sys_path: tuple[str, ...],
) -> frozenset[str]:
    """Prefix-allowlist for paths that may be read when snapshotting stack traces (SAST CWE-22).

    Paths come from Python stack frames captured while generating a tt-metal memory report on
    this machine; they are not arbitrary remote input. Reads are still limited to resolved paths
    under these roots so a crafted trace cannot point at unrelated sensitive files.
    """
    roots: list[str] = []

    def add(raw: str | None) -> None:
        if not raw:
            return
        resolved = _realpath(raw)
        if resolved is None or resolved == os.sep:
            # Never allow the filesystem root as a prefix; it would match every absolute path.
            return
        if resolved not in roots:
            roots.append(resolved)

    add(cwd)
    try:
        add(str(Path.home()))
    except RuntimeError:
        # Best-effort allowlist construction: some environments cannot resolve a home directory.
        pass
    add(sys_prefix)
    if sys_base_prefix != sys_prefix:
        add(sys_base_prefix)
    exe = getattr(sys, "executable", None)
    if exe:
        add(os.path.dirname(exe))
    if venv:
        add(venv)
    try:
        for d in site.getsitepackages():
            add(d)
    except (AttributeError, OSError):
        # Best-effort only: this API may be unavailable or fail on some Python/runtime setups.
        pass
    try:
        add(site.getusersitepackages())
    except (AttributeError, OSError):
        pass
    add(tempfile.gettempdir())
    for entry in sys_path:
        if not entry:
            continue
        add(entry)
    here = os.path.dirname(os.path.abspath(__file__))
    p = here
    for _ in range(8):
        if p == os.sep:
            break
        add(p)
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent

    return frozenset(roots)


def _resolved_path_allowed_for_stack_trace_read(resolved: str) -> bool:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv = _realpath(venv)
    roots = _allowed_stack_trace_source_roots(
        _realpath(os.getcwd()) or os.getcwd(),
        _realpath(sys.prefix) or sys.prefix,
        _realpath(getattr(sys, "base_prefix", sys.prefix)) or getattr(sys, "base_prefix", sys.prefix),
        venv,
        tuple(sys.path),
    )
    return any(_path_is_under_root(resolved, r) for r in roots)


# Gets the first file path from a stack trace
def extract_stack_trace_file(stack_trace_text: str | None) -> str | None:
    if not stack_trace_text:
        return None
    for line in stack_trace_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _PYTHON_STACK_FILE_PATTERN.match(line)
        if match:
            return match.group(1)
        match = _PATH_WITH_LINE_PATTERN.match(line)
        if match:
            return match.group(1)
    return None


def normalize_source_path_from_stack_trace(stack_trace_text: str | None) -> str | None:
    """First source path named in *stack_trace_text*, if it exists on disk and passes read checks."""
    return normalize_existing_source_file_path(extract_stack_trace_file(stack_trace_text))


def normalize_existing_source_file_path(file_path: str | None) -> str | None:
    """Resolve to an absolute path of an existing regular file, or None.

    Stack trace paths are produced by Python while recording a memory report; they are still
    validated (realpath, regular file, prefix allowlist) before any read.
    """
    if not file_path or not isinstance(file_path, str):
        return None
    if "\x00" in file_path:
        return None
    expanded = os.path.expanduser(file_path.strip())
    if not expanded:
        return None
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(os.path.join(os.getcwd(), expanded))
    resolved = _realpath(expanded)
    if resolved is None or not _is_regular_file(resolved):
        return None
    if not _resolved_path_allowed_for_stack_trace_read(resolved):
        return None
    return resolved


def read_source_file(normalized_path: str) -> str | None:
    """Read UTF-8 text from disk for a stack-trace source path.

    Always runs :func:`normalize_existing_source_file_path` before ``open`` (realpath,
    regular-file check, prefix allowlist) so reads never use a raw path string alone.
    """
    if not normalized_path:
        return None
    safe_path = normalize_existing_source_file_path(normalized_path)
    if safe_path is None:
        return None
    try:
        with open(safe_path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def get_source_file_id(cursor: sqlite3.Cursor, path: str, contents: str) -> int:
    """Insert or ignore (path, contents), then return source_files.id for path."""
    cursor.execute(
        "INSERT OR IGNORE INTO source_files (path, contents) VALUES (?, ?)",
        (path, contents),
    )
    cursor.execute("SELECT id FROM source_files WHERE path = ?", (path,))
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"source_files row missing after insert for path {path!r}")
    return row[0]
