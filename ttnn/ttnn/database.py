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


def get_or_create_sqlite_db(report_path):
    global SQLITE_CONNECTION
    sqlite_db_path = report_path / SQLITE_DB_PATH

    if SQLITE_CONNECTION is not None:
        return SQLITE_CONNECTION

    report_path.mkdir(parents=True, exist_ok=True)
    config_path = report_path / CONFIG_PATH
    if not config_path.exists():
        ttnn.save_config_to_json_file(config_path)
    SQLITE_CONNECTION = sqlite_connection = sqlite3.connect(sqlite_db_path)

    logger.debug(f"Creating reports path at {report_path} and sqlite database at {sqlite_db_path}.")

    cursor = sqlite_connection.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS devices
                (
                    device_id int,
                    num_y_cores int,
                    num_x_cores int,
                    num_y_compute_cores int,
                    num_x_compute_cores int,
                    worker_l1_size int,
                    l1_num_banks int,
                    l1_bank_size int,
                    address_at_first_l1_bank int,
                    address_at_first_l1_cb_buffer int,
                    num_banks_per_storage_core int,
                    num_compute_cores int,
                    num_storage_cores int,
                    total_l1_memory int,
                    total_l1_for_tensors int,
                    total_l1_for_interleaved_buffers int,
                    total_l1_for_sharded_buffers int,
                    cb_limit int
                )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS tensors
                (tensor_id int UNIQUE, shape text, dtype text, layout text, memory_config text, device_id int, address int, buffer_type int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS device_tensors
                (tensor_id int, device_id int, address int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS local_tensor_comparison_records
                (tensor_id int UNIQUE, golden_tensor_id int, matches bool, desired_pcc bool, actual_pcc float)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS global_tensor_comparison_records
                (tensor_id int UNIQUE, golden_tensor_id int, matches bool, desired_pcc bool, actual_pcc float)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS operations
                (operation_id int UNIQUE, name text, duration float)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS operation_arguments
                (operation_id int, name text, value text)"""
    )
    cursor.execute(CREATE_SOURCE_FILES_TABLE_SQL)
    cursor.execute(CREATE_STACK_TRACES_TABLE_WITH_SOURCE_SQL)
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS input_tensors
                (operation_id int, input_index int, tensor_id int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS output_tensors
                (operation_id int, output_index int, tensor_id int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS buffers
                (operation_id int, device_id int, address int, max_size_per_bank int, buffer_type int, buffer_layout int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS buffer_pages
                (operation_id int, device_id int, address int, core_y int, core_x int, bank_id int, page_index int, page_address int, page_size int, buffer_type int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS nodes
                (operation_id int, unique_id int, node_operation_id int, name text)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS edges
                (operation_id int, source_unique_id int, sink_unique_id int, source_output_index int, sink_input_index int, key int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS captured_graph
                (operation_id int, captured_graph text)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS errors
                (operation_id int, operation_name text, error_type text, error_message text, stack_trace text, timestamp text)"""
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffers_address ON buffers (address)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffers_operation_id ON buffers (operation_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffers_device_id ON buffers (device_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffers_max_size_per_bank ON buffers (max_size_per_bank)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_buffers_buffer_type ON buffers (buffer_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_output_tensors_tensor_id ON output_tensors (tensor_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_tensors_tensor_id ON input_tensors (tensor_id)")
    cursor.execute(CREATE_INDEX_STACK_TRACES_SOURCE_FILE_SQL)
    sqlite_connection.commit()
    return sqlite_connection


def insert_stack_trace(report_path, operation_id, stack_trace):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

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
