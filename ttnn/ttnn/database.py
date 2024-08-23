# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import sqlite3
import json

from loguru import logger
import networkx as nx

import ttnn

SQLITE_DB_PATH = "db.sqlite"
TENSORS_PATH = "tensors"
GRAPHS_PATH = "graph"
CONFIG_PATH = "config.json"

SQLITE_CONNECTION = None


@dataclasses.dataclass
class Device:
    device_id: int
    num_y_cores: int
    num_x_cores: int
    num_y_compute_cores: int
    num_x_compute_cores: int
    worker_l1_size: int
    l1_num_banks: int
    l1_bank_size: int
    address_at_first_l1_bank: int
    address_at_first_l1_cb_buffer: int
    num_banks_per_storage_core: int
    num_compute_cores: int
    num_storage_cores: int
    total_l1_memory: int
    total_l1_for_tensors: int
    total_l1_for_interleaved_buffers: int
    total_l1_for_sharded_buffers: int
    cb_limit: int


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
class Tensor:
    tensor_id: int
    shape: str
    dtype: str
    layout: str
    memory_config: str
    device_id: int
    address: int
    buffer_type: ttnn.BufferType

    def __post_init__(self):
        self.buffer_type = ttnn.BufferType(self.buffer_type) if self.buffer_type is not None else None


@dataclasses.dataclass
class InputTensor:
    operation_id: int
    input_index: int
    tensor_id: int


@dataclasses.dataclass
class OutputTensor:
    operation_id: int
    output_index: int
    tensor_id: int


@dataclasses.dataclass
class TensorComparisonRecord:
    tensor_id: int
    golden_tensor_id: int
    matches: bool
    desired_pcc: bool
    actual_pcc: float


@dataclasses.dataclass
class OperationArgument:
    operation_id: int
    name: str
    value: str


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
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS stack_traces
                (operation_id int, stack_trace text)"""
    )
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
                (operation_id int, device_id int, address int, max_size_per_bank int, buffer_type int)"""
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
    sqlite_connection.commit()
    return sqlite_connection


def get_cursor(report_path):
    sqlite_connection = get_or_create_sqlite_db(report_path)
    return sqlite_connection.cursor()


DEVICE_IDS_IN_DATABASE = set()


def insert_devices(report_path, devices):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    for device in devices:
        if device.id() in DEVICE_IDS_IN_DATABASE:
            continue
        device_info = ttnn._ttnn.reports.get_device_info(device)
        cursor.execute(
            f"""INSERT INTO devices VALUES (
                {device.id()},
                {device_info.num_y_cores},
                {device_info.num_x_cores},
                {device_info.num_y_compute_cores},
                {device_info.num_x_compute_cores},
                {device_info.worker_l1_size},
                {device_info.l1_num_banks},
                {device_info.l1_bank_size},
                {device_info.address_at_first_l1_bank},
                {device_info.address_at_first_l1_cb_buffer},
                {device_info.num_banks_per_storage_core},
                {device_info.num_compute_cores},
                {device_info.num_storage_cores},
                {device_info.total_l1_memory},
                {device_info.total_l1_for_tensors},
                {device_info.total_l1_for_interleaved_buffers},
                {device_info.total_l1_for_sharded_buffers},
                {device_info.cb_limit}
            )"""
        )
        sqlite_connection.commit()
        DEVICE_IDS_IN_DATABASE.add(device.id())


def optional_value(value, text=False):
    if value is None:
        return "NULL"
    if text:
        return f"'{value}'"
    return value


def insert_operation(report_path, operation_id, operation, duration):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute(
        f"""INSERT INTO operations VALUES (
            {operation_id}, '{operation.python_fully_qualified_name}', {optional_value(duration)}
            )
        ON CONFLICT (operation_id)
        DO UPDATE
        SET
            duration=EXCLUDED.duration;"""
    )
    sqlite_connection.commit()


def insert_stack_trace(report_path, operation_id, stack_trace):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    formatted_stack_trace = "\n".join(stack_trace[:-2][::-1])

    cursor.execute(f"INSERT INTO stack_traces VALUES ({operation_id}, '{formatted_stack_trace}')")
    sqlite_connection.commit()


def insert_tensor(report_path, tensor):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    if query_tensor_by_id(report_path, tensor.tensor_id) is not None:
        return

    if ttnn.has_storage_type_of(tensor, ttnn.DEVICE_STORAGE_TYPE) and tensor.is_allocated():
        address = tensor.buffer_address()
        device_id = tensor.device().id()
        memory_config = ttnn.get_memory_config(tensor)
        buffer_type = memory_config.buffer_type.value
    else:
        address = None
        device_id = None
        memory_config = None
        buffer_type = None

    cursor.execute(
        f"""
        INSERT INTO tensors VALUES (
            {tensor.tensor_id},
            '{tensor.shape}',
            '{tensor.dtype}',
            '{tensor.layout}',
            {optional_value(memory_config, text=True)},
            {optional_value(device_id)},
            {optional_value(address)},
            {optional_value(buffer_type)})"""
    )
    sqlite_connection.commit()


def insert_input_tensors(report_path, operation_id, input_tensors):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    for input_index, tensor in enumerate(input_tensors):
        insert_tensor(report_path, tensor)

        cursor.execute(
            f"""INSERT INTO input_tensors VALUES (
                {operation_id},
                {input_index},
                {tensor.tensor_id}
            )"""
        )
    sqlite_connection.commit()

    if ttnn.CONFIG.enable_detailed_tensor_report:
        for tensor in input_tensors:
            store_tensor(report_path, tensor)


def insert_output_tensors(report_path, operation_id, output_tensors):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    for output_index, tensor in enumerate(output_tensors):
        insert_tensor(report_path, tensor)

        cursor.execute(
            f"""INSERT INTO output_tensors VALUES (
                {operation_id},
                {output_index},
                {tensor.tensor_id}
            )"""
        )
    sqlite_connection.commit()

    if ttnn.CONFIG.enable_detailed_tensor_report:
        for tensor in output_tensors:
            store_tensor(report_path, tensor)


def insert_buffers(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    for buffer in ttnn._ttnn.reports.get_buffers():
        cursor.execute(
            f"""INSERT INTO buffers VALUES (
                {operation_id},
                {buffer.device_id},
                {buffer.address},
                {buffer.max_size_per_bank},
                {buffer.buffer_type.value}
            )"""
        )
    sqlite_connection.commit()


def insert_buffer_pages(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()
    for buffer_page in ttnn._ttnn.reports.get_buffer_pages():
        cursor.execute(
            f"""INSERT INTO buffer_pages VALUES (
                {operation_id},
                {buffer_page.device_id},
                {buffer_page.address},
                {buffer_page.core_y},
                {buffer_page.core_x},
                {buffer_page.bank_id},
                {buffer_page.page_index},
                {buffer_page.page_address},
                {buffer_page.page_size},
                {buffer_page.buffer_type.value}
            )"""
        )
    sqlite_connection.commit()


def store_graph(report_path, operation_id, graph):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    for node in graph.nodes:
        node_attributes = graph.nodes[node]
        node_operation = node_attributes["operation"]
        node_name = str(node_operation)
        node_operation_id = node_attributes.get("operation_id", None)
        cursor.execute(
            f"""INSERT INTO nodes VALUES (
                {operation_id},
                {node.unique_id},
                {optional_value(node_operation_id)},
                '{node_name}'
            )"""
        )
    for source_node, sink_node, key, data in graph.edges(keys=True, data=True):
        cursor.execute(
            f"""INSERT INTO edges VALUES (
                {operation_id},
                {source_node.unique_id},
                {sink_node.unique_id},
                {data['source_output_index']},
                {data['sink_input_index']},
                {key}
            )"""
        )

    sqlite_connection.commit()


def load_graph(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    graph = nx.MultiDiGraph()
    cursor.execute("SELECT * FROM nodes WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        _, unique_id, node_operation_id, node_name = row
        graph.add_node(unique_id, node_operation_id=node_operation_id, name=node_name)

    cursor.execute("SELECT * FROM edges WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        _, source_unique_id, sink_unique_id, source_output_index, sink_input_index, key = row
        graph.add_edge(
            source_unique_id,
            sink_unique_id,
            source_output_index=source_output_index,
            sink_input_index=sink_input_index,
            key=key,
        )

    return graph


def store_tensor(report_path, tensor):
    import torch

    tensors_path = report_path / TENSORS_PATH
    tensors_path.mkdir(parents=True, exist_ok=True)
    if isinstance(tensor, ttnn.Tensor):
        tensor_file_name = tensors_path / f"{tensor.tensor_id}.bin"
        if tensor_file_name.exists():
            return
        ttnn.dump_tensor(
            tensor_file_name,
            ttnn.from_device(tensor),
        )
    elif isinstance(tensor, torch.Tensor):
        tensor_file_name = tensors_path / f"{tensor.tensor_id}.pt"
        if tensor_file_name.exists():
            return
        torch.save(torch.Tensor(tensor), tensor_file_name)
    else:
        raise ValueError(f"Unsupported tensor type {type(tensor)}")


def insert_captured_graph(report_path, operation_id, captured_graph):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute(
        f"""INSERT INTO captured_graph VALUES (
            {operation_id},
            '{json.dumps(captured_graph)}'
        )"""
    )
    sqlite_connection.commit()


def get_tensor_file_name_by_id(report_path, tensor_id):
    tensors_path = report_path / TENSORS_PATH
    tensors_path.mkdir(parents=True, exist_ok=True)
    tensor_path = tensors_path / f"{tensor_id}.bin"
    if tensor_path.exists():
        return tensor_path
    tensor_path = tensors_path / f"{tensor_id}.pt"
    if tensor_path.exists():
        return tensor_path
    return None


def load_tensor_by_id(report_path, tensor_id, device=None):
    import torch

    tensors_path = report_path / TENSORS_PATH
    tensors_path.mkdir(parents=True, exist_ok=True)
    tensor_path = tensors_path / f"{tensor_id}.bin"
    if tensor_path.exists():
        return ttnn.load_tensor(tensor_path, device=device)
    tensor_path = tensors_path / f"{tensor_id}.pt"
    if tensor_path.exists():
        return torch.load(tensor_path)
    return None


def convert_arguments_to_strings(function_args, function_kwargs):
    import torch

    index = 0

    def recursive_preprocess_golden_function_inputs(object):
        nonlocal index
        if isinstance(object, (ttnn.Tensor, torch.Tensor)):
            output = f"{object}"
            index += 1
            return output
        elif isinstance(object, (list, tuple)):
            new_object = [recursive_preprocess_golden_function_inputs(element) for element in object]
            return ", ".join(new_object)
        else:
            return f"{object}"

    new_args = []
    for arg in function_args:
        new_arg = recursive_preprocess_golden_function_inputs(arg)
        new_args.append(new_arg)
    new_kwargs = {}
    for key, value in function_kwargs.items():
        new_value = recursive_preprocess_golden_function_inputs(value)
        new_kwargs[key] = new_value
    return new_args, new_kwargs


def insert_operation_arguments(report_path, operation_id, function_args, function_kwargs):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    function_args, function_kwargs = convert_arguments_to_strings(function_args, function_kwargs)

    for index, arg in enumerate(function_args):
        cursor.execute(
            f"""INSERT INTO operation_arguments VALUES (
                {operation_id},
                '{index}',
                '{arg}'
            )"""
        )
    for key, value in function_kwargs.items():
        cursor.execute(
            f"""INSERT INTO operation_arguments VALUES (
                {operation_id},
                '{key}',
                '{value}'
            )"""
        )
    sqlite_connection.commit()


def insert_tensor_comparison_records(report_path, table_name, tensor_comparison_records):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    for record in tensor_comparison_records:
        cursor.execute(
            f"""INSERT INTO {table_name} VALUES (
                {record.tensor_id},
                {record.golden_tensor_id},
                {record.matches},
                {record.desired_pcc},
                {record.actual_pcc}
            )"""
        )
    sqlite_connection.commit()


def store_tensors(report_path, tensors):
    for tensor in tensors:
        insert_tensor(report_path, tensor)
        if ttnn.CONFIG.enable_detailed_tensor_report:
            store_tensor(report_path, tensor)


def query_device_by_id(report_path, device_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM devices WHERE device_id = ?", (device_id,))
    device = None
    for row in cursor.fetchall():
        device = ttnn.database.Device(*row)
        break

    return device


def query_operation_by_id(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations WHERE operation_id = ?", (operation_id,))
    operation = None
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        break

    return operation


def query_operation_by_id_together_with_previous_and_next(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations WHERE operation_id = ?", (operation_id,))
    operation = None
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        break

    cursor.execute(
        "SELECT * FROM operations WHERE operation_id < ? ORDER BY operation_id DESC LIMIT 1", (operation_id,)
    )
    previous_operation = None
    for row in cursor.fetchall():
        previous_operation = ttnn.database.Operation(*row)
        break

    cursor.execute("SELECT * FROM operations WHERE operation_id > ? ORDER BY operation_id ASC LIMIT 1", (operation_id,))
    next_operation = None
    for row in cursor.fetchall():
        next_operation = ttnn.database.Operation(*row)
        break

    return operation, previous_operation, next_operation


def query_operations(report_path):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations")
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        yield operation


def query_latest_operation(report_path):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations ORDER BY operation_id DESC LIMIT 1")
    operation = None
    for row in cursor.fetchall():
        operation = ttnn.database.Operation(*row)
        break

    return operation


def query_operation_arguments(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operation_arguments WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        operation_argument = ttnn.database.OperationArgument(*row)
        yield operation_argument


def query_stack_trace(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM stack_traces WHERE operation_id = ?", (operation_id,))
    stack_trace = None
    for row in cursor.fetchall():
        _, stack_trace = row
        break
    return stack_trace


def query_buffers(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM buffers WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        yield ttnn.database.Buffer(*row)


def query_buffer_pages(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM buffer_pages WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        yield ttnn.database.BufferPage(*row)


def query_tensor_by_id(report_path, tensor_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM tensors WHERE tensor_id = ?", (tensor_id,))
    tensor = None
    for row in cursor.fetchall():
        tensor = ttnn.database.Tensor(*row)
        break

    return tensor


def query_latest_tensor(report_path):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM tensors ORDER BY tensor_id DESC LIMIT 1")
    tensor = None
    for row in cursor.fetchall():
        tensor = ttnn.database.Tensor(*row)
        break

    return tensor


def query_input_tensors(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM input_tensors WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        yield ttnn.database.InputTensor(*row)


def query_output_tensors(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM output_tensors WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        yield ttnn.database.OutputTensor(*row)


def query_output_tensor_by_tensor_id(report_path, tensor_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM output_tensors WHERE tensor_id = ?", (tensor_id,))
    output_tensor = None
    for row in cursor.fetchall():
        output_tensor = ttnn.database.OutputTensor(*row)
        break

    return output_tensor


def query_tensor_comparison_record(report_path, table_name, tensor_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute(f"SELECT * FROM {table_name} WHERE tensor_id = ?", (tensor_id,))
    tensor_comparison_record = None
    for row in cursor.fetchall():
        tensor_comparison_record = ttnn.database.TensorComparisonRecord(*row)
        break

    return tensor_comparison_record


def query_producer_operation_id(report_path, tensor_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM output_tensors WHERE tensor_id = ?", (tensor_id,))
    operation_id = None
    for row in cursor.fetchall():
        operation_id, *_ = row
        break

    return operation_id


def query_consumer_operation_ids(report_path, tensor_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM input_tensors WHERE tensor_id = ?", (tensor_id,))
    for row in cursor.fetchall():
        operation_id, *_ = row
        yield operation_id


def query_captured_graph(report_path, operation_id):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db(report_path)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM captured_graph WHERE operation_id = ?", (operation_id,))
    captured_graph = None
    for row in cursor.fetchall():
        _, captured_graph = row
        break
    return json.loads(captured_graph)
