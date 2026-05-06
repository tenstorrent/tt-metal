# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for resolving Python stack traces to on-disk source and SQLite source_files rows."""

import re
import sqlite3
from pathlib import Path

_PYTHON_STACK_FILE_PATTERN = re.compile(r'^\s*File "([^"]+)", line \d+, in ')
_PATH_WITH_LINE_PATTERN = re.compile(r"^\s*([^:\n]+):(\d+)(?::\d+)?\s*$")


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


def normalize_existing_source_file_path(file_path: str | None) -> str | None:
    if not file_path:
        return None

    candidate = Path(file_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)

    if not candidate.is_file():
        return None

    return str(candidate.resolve(strict=False))


def read_source_file_contents(file_path: str) -> str | None:
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def insert_source_file_id_column(cursor: sqlite3.Cursor) -> None:
    """Add ``source_file_id`` to ``stack_traces`` when upgrading a pre-existing database.

    ``CREATE TABLE IF NOT EXISTS`` does not alter legacy two-column ``stack_traces`` tables;
    without this step, new three-value inserts and indexes on ``source_file_id`` would fail.
    """
    cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='stack_traces'")
    if cursor.fetchone() is None:
        return
    cursor.execute("PRAGMA table_info(stack_traces)")
    column_names = {row[1] for row in cursor.fetchall()}
    if "source_file_id" in column_names:
        return
    cursor.execute("ALTER TABLE stack_traces ADD COLUMN source_file_id int REFERENCES source_files(id)")


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


# ``ttnn.database`` / ``graph_report`` import ``get_source_file_id`` by this exact name.
# Keep typo/historical aliases so mixed revisions or stale wheels do not break imports.
ensure_source_file_id = get_source_file_id
get_source_file_id_id = get_source_file_id
