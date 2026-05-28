# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import sqlite3

from loguru import logger

import ttnn
from .stack_trace_source import (
    CREATE_INDEX_STACK_TRACES_SOURCE_FILE_SQL,
    CREATE_SOURCE_FILES_TABLE_SQL,
    CREATE_STACK_TRACES_TABLE_WITH_SOURCE_SQL,
    get_source_file_id,
    normalize_source_path_from_stack_trace,
    read_source_file,
)

SQLITE_DB_PATH = "db.sqlite"
CONFIG_PATH = "config.json"

SQLITE_CONNECTION = None


@dataclasses.dataclass
class Operation:
    operation_id: int
    name: str
    duration: float


@dataclasses.dataclass
class Buffer:
    operation_id: int
    device_id: int
    address: int
    max_size_per_bank: int
    buffer_type: ttnn.BufferType
    buffer_layout: ttnn.TensorMemoryLayout

    def __post_init__(self):
        self.buffer_type = ttnn.BufferType(self.buffer_type) if self.buffer_type is not None else None


@dataclasses.dataclass
class BufferPage:
    operation_id: int
    device_id: int
    address: int
    core_y: int
    core_x: int
    bank_id: int
    page_index: int
    page_address: int
    page_size: int
    buffer_type: ttnn.BufferType

    def __post_init__(self):
        self.buffer_type = ttnn.BufferType(self.buffer_type) if self.buffer_type is not None else None


@dataclasses.dataclass
class ErrorRecord:
    operation_id: int
    operation_name: str
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: str


def insert_stack_trace(report_path, operation_id, stack_trace):
    global SQLITE_CONNECTION
    if SQLITE_CONNECTION is None:
        report_path.mkdir(parents=True, exist_ok=True)
        config_path = report_path / CONFIG_PATH
        if not config_path.exists():
            ttnn.save_config_to_json_file(config_path)
        sqlite_db_path = report_path / SQLITE_DB_PATH
        SQLITE_CONNECTION = sqlite3.connect(sqlite_db_path)
        logger.debug(f"Creating reports path at {report_path} and sqlite database at {sqlite_db_path}.")
    sqlite_connection = SQLITE_CONNECTION
    cursor = sqlite_connection.cursor()
    cursor.execute(CREATE_SOURCE_FILES_TABLE_SQL)
    cursor.execute(CREATE_STACK_TRACES_TABLE_WITH_SOURCE_SQL)
    cursor.execute(CREATE_INDEX_STACK_TRACES_SOURCE_FILE_SQL)

    formatted_stack_trace = "\n".join(stack_trace[:-2][::-1])

    # let sqlite handle formatting strings with mixed quotes
    source_file_id = None
    normalized_path = normalize_source_path_from_stack_trace(formatted_stack_trace)
    if normalized_path is not None:
        file_contents = read_source_file(normalized_path)
        if file_contents is not None:
            source_file_id = get_source_file_id(cursor, normalized_path, file_contents)

    statement = "INSERT INTO stack_traces (operation_id, stack_trace, source_file_id) VALUES (?, ?, ?)"
    cursor.execute(statement, (operation_id, formatted_stack_trace, source_file_id))

    sqlite_connection.commit()
