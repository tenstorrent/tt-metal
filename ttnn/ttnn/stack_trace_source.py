# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for resolving Python stack traces to on-disk source and SQLite source_files rows."""

import re
import sqlite3
from pathlib import Path

_PYTHON_STACK_FILE_PATTERN = re.compile(r'^\s*File "([^"]+)", line \d+, in ')
_PATH_WITH_LINE_PATTERN = re.compile(r"^\s*([^:\n]+):(\d+)(?::\d+)?\s*$")


def extract_last_stack_trace_file(stack_trace_text: str | None) -> str | None:
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


def ensure_source_file_id(cursor: sqlite3.Cursor, path: str, contents: str) -> int:
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
