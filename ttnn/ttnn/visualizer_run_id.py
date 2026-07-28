# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy-free run_id helpers shared by TTNN memory reports and Tracy perf reports."""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path
from typing import MutableMapping, Optional

from loguru import logger

TT_METAL_RUN_ID_ENV = "TT_METAL_RUN_ID"
RUN_ID_METADATA_KEY = "run_id"


def peek_run_id() -> Optional[str]:
    """Return ``TT_METAL_RUN_ID`` if set and non-blank; do not mint."""
    value = os.environ.get(TT_METAL_RUN_ID_ENV, "").strip()
    return value or None


def get_or_create_run_id() -> str:
    """Return ``TT_METAL_RUN_ID``, minting and exporting one if unset."""
    existing = peek_run_id()
    if existing:
        return existing
    run_id = str(uuid.uuid4())
    os.environ[TT_METAL_RUN_ID_ENV] = run_id
    return run_id


def inject_run_id_into_env(env: MutableMapping[str, str]) -> str:
    """Mint/reuse run_id and write it into ``env`` for a child process.

    ``env`` is typically a snapshot of ``os.environ`` taken before minting; the
    parent process ``os.environ`` is updated via ``get_or_create_run_id`` as well.
    """
    run_id = get_or_create_run_id()
    env[TT_METAL_RUN_ID_ENV] = run_id
    return run_id


def read_db_run_id(db_path: Path | str) -> Optional[str]:
    """Return ``report_metadata`` run_id from a memory ``db.sqlite``, if present.

    Does not create the database file when it is missing.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        return None

    try:
        # Resolve so relative paths (e.g. from DEFAULT_TTNN_REPORTS_ROOT) work;
        # Path.as_uri() requires an absolute path. URI mode=ro refuses to create
        # a missing file (defence in depth).
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
        
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM report_metadata WHERE key = ?",
                (RUN_ID_METADATA_KEY,),
            )
            row = cursor.fetchone()
            return str(row[0]) if row and row[0] else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def stamp_memory_run_id(db_path: Path | str) -> Optional[str]:
    """Insert ``run_id`` into ``report_metadata`` of a memory ``db.sqlite``."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return None

    run_id = get_or_create_run_id()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO report_metadata (key, value) VALUES (?, ?)",
            (RUN_ID_METADATA_KEY, run_id),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Visualizer run_id={run_id} written to {db_path} report_metadata")
    return run_id


def stamp_report_dir_run_id(report_path: Path | str, *, is_primary_rank: bool = True) -> Optional[str]:
    """Stamp ``run_id`` into ``report_path/db.sqlite`` on the primary rank when the DB exists.

    Used by the pytest graph-report fixture so memory reports pair with Tracy manifests.
    """
    if not is_primary_rank:
        return None
    return stamp_memory_run_id(Path(report_path) / "db.sqlite")
