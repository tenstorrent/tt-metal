# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""
Import C++ graph capture reports into ttnn-visualizer SQLite database.

This module completely decouples graph capture (C++) from visualization (SQLite).
No database operations happen during model execution - everything is offline.

Workflow:
    1. C++ captures graph to JSON: ttnn::graph::end_graph_capture_to_file("report.json")
    2. Later, import to SQLite: python -m ttnn.graph_report report.json ./visualizer_db/
    3. Open ttnn-visualizer pointing to ./visualizer_db/

This replaces the invasive approach where decorators.py inserted into SQLite during execution.

Note: Comparison mode (golden tensor validation) is Python-specific and still writes
directly to SQLite during execution. The importer is aware of this and uses
CREATE TABLE IF NOT EXISTS to avoid conflicts with comparison mode data.
"""

import json
import math
import sqlite3
from pathlib import Path
from typing import Union

SUPPORTED_REPORT_VERSION = 1


def create_database_schema(cursor: sqlite3.Cursor) -> None:
    """Create the full ttnn-visualizer SQLite database schema."""

    # Devices table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS devices (
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
        )
    """
    )

    # Operations table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS operations (
            operation_id int UNIQUE,
            name text,
            duration float
        )
    """
    )

    # Operation arguments
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS operation_arguments (
            operation_id int,
            name text,
            value text
        )
    """
    )

    # Tensors table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tensors (
            tensor_id int UNIQUE,
            shape text,
            dtype text,
            layout text,
            memory_config text,
            device_id int,
            address int,
            buffer_type int
        )
    """
    )

    # Device tensors table (for multi-device tensor placement)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS device_tensors (
            tensor_id int,
            device_id int,
            address int
        )
    """
    )

    # Buffers table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS buffers (
            operation_id int,
            device_id int,
            address int,
            max_size_per_bank int,
            buffer_type int,
            buffer_layout int
        )
    """
    )

    # Captured graph (raw JSON)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS captured_graph (
            operation_id int,
            captured_graph text
        )
    """
    )

    # Graph nodes (extracted from captured_graph)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            operation_id int,
            unique_id int,
            node_operation_id int,
            name text
        )
    """
    )

    # Graph edges (extracted from connections in captured_graph)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            operation_id int,
            source_unique_id int,
            sink_unique_id int,
            source_output_index int,
            sink_input_index int,
            key int
        )
    """
    )

    # Report metadata
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS report_metadata (
            key text UNIQUE,
            value text
        )
    """
    )

    # Errors table (populated from C++ error nodes)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS errors (
            operation_id int,
            error_type text,
            error_message text,
            error_operation text
        )
    """
    )

    # Stack traces table (when stack trace capture is enabled)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stack_traces (
            operation_id int,
            stack_trace text
        )
    """
    )

    # Input/output tensors
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS input_tensors (
            operation_id int,
            input_index int,
            tensor_id int
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS output_tensors (
            operation_id int,
            output_index int,
            tensor_id int
        )
    """
    )

    # Comparison mode tables (populated by Python runtime, not importer)
    # These are created here for schema completeness but data comes from
    # ttnn.database when comparison mode is enabled during execution
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS local_tensor_comparison_records (
            tensor_id int,
            golden_tensor_id int,
            matches int,
            desired_pcc float,
            actual_pcc float
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS global_tensor_comparison_records (
            tensor_id int,
            golden_tensor_id int,
            matches int,
            desired_pcc float,
            actual_pcc float
        )
    """
    )

    # Buffer pages table (when detailed buffer report is enabled)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS buffer_pages (
            operation_id int,
            device_id int,
            address int,
            core_y int,
            core_x int,
            bank_id int,
            page_index int,
            page_address int,
            page_size int,
            buffer_type int
        )
    """
    )


def import_devices(cursor: sqlite3.Cursor, devices: list) -> set:
    """Import device information using batch insert. Returns set of imported device IDs."""
    if not devices:
        return set()

    batch = []
    imported_ids = set()
    for device in devices:
        device_id = device.get("device_id", 0)
        batch.append(
            (
                device_id,
                device.get("num_y_cores", 0),
                device.get("num_x_cores", 0),
                device.get("num_y_compute_cores", 0),
                device.get("num_x_compute_cores", 0),
                device.get("worker_l1_size", 0),
                device.get("l1_num_banks", 0),
                device.get("l1_bank_size", 0),
                device.get("address_at_first_l1_bank", 0),
                device.get("address_at_first_l1_cb_buffer", 0),
                device.get("num_banks_per_storage_core", 0),
                device.get("num_compute_cores", 0),
                device.get("num_storage_cores", 0),
                device.get("total_l1_memory", 0),
                device.get("total_l1_for_tensors", 0),
                device.get("total_l1_for_interleaved_buffers", 0),
                device.get("total_l1_for_sharded_buffers", 0),
                device.get("cb_limit", 0),
            )
        )
        imported_ids.add(device_id)

    cursor.executemany(
        """INSERT OR REPLACE INTO devices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", batch
    )
    return imported_ids


def _compute_max_size_per_bank(size, page_size, buffer_type, layout, num_cores, dev_info):
    """Compute per-bank buffer allocation from total size and device bank count."""
    if page_size <= 0 or size <= 0:
        return size

    total_pages = size // page_size

    if layout == "INTERLEAVED":
        num_banks = dev_info.get("num_dram_channels", 0) if buffer_type == 0 else dev_info.get("l1_num_banks", 0)
        if num_banks > 0:
            pages_per_bank = math.ceil(total_pages / num_banks)
            return pages_per_bank * page_size
    elif num_cores > 0:
        pages_per_core = math.ceil(total_pages / num_cores)
        return pages_per_core * page_size

    return size


def _validate_graph_integrity(
    operations_batch,
    tensors_batch,
    input_tensors_batch,
    output_tensors_batch,
    operation_arguments_batch,
    device_tensors_batch,
    buffers_batch,
    devices,
) -> list:
    """
    Validate referential integrity of imported graph data.
    Returns a list of warning strings for any violations found.
    """
    warnings = []

    operation_ids = {row[0] for row in operations_batch}
    tensor_ids = {int(row[0]) if isinstance(row[0], str) else row[0] for row in tensors_batch}
    device_ids = {dev.get("device_id", 0) for dev in devices} if devices else set()

    # input_tensors.tensor_id must reference a known tensor
    for op_id, idx, tid in input_tensors_batch:
        tid_int = int(tid) if isinstance(tid, str) else tid
        if tid_int not in tensor_ids:
            warnings.append(
                f"input_tensors references tensor_id={tid} (operation_id={op_id}, input_index={idx}) "
                f"which does not exist in tensors table. "
                f"This may indicate node counters are being stored instead of real tensor_ids."
            )

    # output_tensors.tensor_id must reference a known tensor
    for op_id, idx, tid in output_tensors_batch:
        tid_int = int(tid) if isinstance(tid, str) else tid
        if tid_int not in tensor_ids:
            warnings.append(
                f"output_tensors references tensor_id={tid} (operation_id={op_id}, output_index={idx}) "
                f"which does not exist in tensors table."
            )

    # input/output_tensors.operation_id must reference a known operation
    for op_id, idx, tid in input_tensors_batch:
        if op_id not in operation_ids:
            warnings.append(f"input_tensors references operation_id={op_id} which does not exist in operations table.")

    for op_id, idx, tid in output_tensors_batch:
        if op_id not in operation_ids:
            warnings.append(f"output_tensors references operation_id={op_id} which does not exist in operations table.")

    # operation_arguments.operation_id must reference a known operation
    for op_id, name, val in operation_arguments_batch:
        if op_id not in operation_ids:
            warnings.append(
                f"operation_arguments references operation_id={op_id} which does not exist in operations table."
            )
            break  # one warning is enough for args

    # device_tensors.tensor_id must reference a known tensor
    for tid, dev_id, addr in device_tensors_batch:
        tid_int = int(tid) if isinstance(tid, str) else tid
        if tid_int not in tensor_ids:
            warnings.append(f"device_tensors references tensor_id={tid} which does not exist in tensors table.")

    # buffers.device_id should reference a known device (if device info available)
    if device_ids:
        bad_device_ids = set()
        for row in buffers_batch:
            dev_id = row[1]
            if dev_id not in device_ids:
                bad_device_ids.add(dev_id)
        for dev_id in bad_device_ids:
            warnings.append(
                f"buffers references device_id={dev_id} which does not exist in devices table. "
                f"Known devices: {sorted(device_ids)}"
            )

    return warnings


def import_graph(
    cursor: sqlite3.Cursor,
    graph: list,
    base_operation_id: int = 0,
    devices: list = None,
    detailed: bool = False,
) -> dict:
    """
    Import graph trace into database using batch inserts for performance.

    Extracts and imports:
    - Operations with durations
    - Tensors
    - Buffers
    - Operation arguments
    - Input/output tensor relationships
    - Graph edges

    Args:
        devices: Raw device info dicts from the report, used to compute per-bank buffer sizes.
        detailed: When False (default), only device tensors are imported and duplicates
            are merged by address to match the legacy Python capture behavior.
            When True, all tensors (host + device) are imported with full detail.

    Returns dict with stats about what was imported.
    """
    # Collect data for batch inserts
    nodes_batch = []
    stack_traces_batch = []
    operations_batch = []
    operation_arguments_batch = []
    input_tensors_batch = []
    output_tensors_batch = []
    tensors_batch = []
    device_tensors_batch = []
    buffers_batch = []
    errors_batch = []
    edges_batch = []
    captured_graph_batch = []

    # Build device bank info for per-bank buffer size computation
    device_bank_info = {}
    if devices:
        for dev in devices:
            dev_id = dev.get("device_id", 0)
            device_bank_info[dev_id] = {
                "num_dram_channels": dev.get("num_dram_channels", 0),
                "l1_num_banks": dev.get("l1_num_banks", 0),
            }

    # Track function start nodes to pair with function end
    function_stack = []
    operation_counter = 0
    tensor_ids_seen = set()
    active_buffers = []
    current_op_nodes = []
    op_nesting_depth = 0

    # First pass: collect all data
    for node in graph:
        node_type = node.get("node_type", "")
        counter = node.get("counter", 0)
        params = node.get("params", {})

        # Collect nodes for per-operation captured_graph subgraphs
        if op_nesting_depth > 0:
            current_op_nodes.append(node)

        if node_type == "function_start":
            name = params.get("name", "unknown")
            is_nested = not detailed and len(function_stack) > 0
            function_stack.append(
                {
                    "counter": counter,
                    "name": name,
                    "arguments": node.get("arguments", []),
                    "input_tensors": node.get("input_tensors", []),
                    "nested": is_nested,
                }
            )

            if not is_nested:
                op_nesting_depth += 1
                current_op_nodes = [node]
                nodes_batch.append((base_operation_id, counter, operation_counter, name))

                stack_trace = node.get("stack_trace", [])
                if stack_trace:
                    stack_traces_batch.append((base_operation_id + operation_counter, "\n".join(stack_trace)))
            else:
                op_nesting_depth += 1

        elif node_type == "function_end":
            name = params.get("name", "unknown")

            start_node = function_stack.pop() if function_stack else None
            if start_node and start_node.get("nested"):
                op_nesting_depth -= 1
                continue

            op_nesting_depth -= 1

            duration_ns = node.get("duration_ns", 0)
            duration_s = duration_ns / 1e9 if duration_ns else 0

            operation_id = base_operation_id + operation_counter
            operations_batch.append((operation_id, name, duration_s))

            if start_node:
                for idx, arg in enumerate(start_node.get("arguments", [])):
                    operation_arguments_batch.append((operation_id, f"arg_{idx}", str(arg)))

                for idx, node_counter in enumerate(start_node.get("input_tensors", [])):
                    if node_counter < len(graph):
                        tensor_node = graph[node_counter]
                        if tensor_node.get("node_type") == "tensor":
                            real_tensor_id = tensor_node.get("params", {}).get("tensor_id", "")
                            if real_tensor_id:
                                input_tensors_batch.append((operation_id, idx, int(real_tensor_id)))

            # Output tensor relationships from connections
            output_tensor_nodes = []
            connections = node.get("connections", [])
            output_idx = 0
            for conn_id in connections:
                if conn_id < len(graph):
                    conn_node = graph[conn_id]
                    if conn_node.get("node_type") == "tensor":
                        tensor_id = conn_node.get("params", {}).get("tensor_id", "")
                        if tensor_id:
                            output_tensors_batch.append((operation_id, output_idx, tensor_id))
                            output_idx += 1
                            output_tensor_nodes.append(conn_node)

            # Per-operation captured_graph subgraph
            capture_start = {
                "arguments": [],
                "connections": [1],
                "counter": 0,
                "input_tensors": [],
                "node_type": "capture_start",
                "params": {},
                "stacking_level": 0,
            }
            capture_end = {
                "arguments": [],
                "connections": [],
                "counter": 0,
                "input_tensors": [],
                "node_type": "capture_end",
                "params": {},
                "stacking_level": 0,
            }
            subgraph = [capture_start] + current_op_nodes + output_tensor_nodes + [capture_end]
            captured_graph_batch.append((operation_id, json.dumps(subgraph)))
            current_op_nodes = []

            # Cumulative buffer snapshot: emit all currently-active buffers for this operation
            for buf in active_buffers:
                buffers_batch.append((operation_id, *buf))

            operation_counter += 1

        elif node_type == "tensor":
            tensor_id = params.get("tensor_id", "")
            if tensor_id and tensor_id not in tensor_ids_seen:
                tensor_ids_seen.add(tensor_id)
                shape = params.get("shape", "")
                dtype = params.get("dtype")
                layout = params.get("layout")
                memory_config = params.get("memory_config")
                device_id = int(params["device_id"]) if "device_id" in params else None
                address = int(params["address"]) if "address" in params else None
                buffer_type_str = params.get("buffer_type")
                buffer_type = None
                if buffer_type_str:
                    buffer_type_map = {"DRAM": 0, "L1": 1, "SYSTEM_MEMORY": 2, "L1_SMALL": 3, "TRACE": 4}
                    buffer_type = buffer_type_map.get(buffer_type_str, 0)

                tensors_batch.append((tensor_id, shape, dtype, layout, memory_config, device_id, address, buffer_type))

                device_tensors_str = params.get("device_tensors")
                if device_tensors_str:
                    device_tensors = json.loads(device_tensors_str)
                    for dt in device_tensors:
                        device_tensors_batch.append(
                            (tensor_id, dt.get("mesh_device_id", dt.get("device_id")), dt.get("address"))
                        )

        elif node_type == "buffer_allocate":
            device_id = int(params.get("device_id", 0))
            address = int(params.get("address", 0))
            size = int(params.get("size", 0))
            page_size = int(params.get("page_size", 0))
            num_cores = int(params.get("num_cores", 0))
            buffer_type = 1 if params.get("type") == "L1" else 0
            layout = params.get("layout", "INTERLEAVED")
            layout_int = {"INTERLEAVED": 0, "HEIGHT_SHARDED": 1, "WIDTH_SHARDED": 2, "BLOCK_SHARDED": 3}.get(layout, 0)

            max_size_per_bank = _compute_max_size_per_bank(
                size,
                page_size,
                buffer_type,
                layout,
                num_cores,
                device_bank_info.get(device_id, {}),
            )
            active_buffers.append((device_id, address, max_size_per_bank, buffer_type, layout_int))

        elif node_type == "buffer_deallocate":
            dealloc_address = int(params.get("address", 0))
            active_buffers = [b for b in active_buffers if b[1] != dealloc_address]

        elif node_type == "error":
            error_type = params.get("error_type", "unknown")
            error_message = params.get("error_message", "")
            error_operation = params.get("error_operation", "")
            errors_batch.append((base_operation_id, error_type, error_message, error_operation))

    # Second pass: collect edges
    sink_input_counts = {}
    edge_key_counts = {}
    for node in graph:
        source_id = node.get("counter", 0)
        connections = node.get("connections", [])

        for output_idx, sink_id in enumerate(connections):
            input_idx = sink_input_counts.get(sink_id, 0)
            sink_input_counts[sink_id] = input_idx + 1

            edge_pair = (source_id, sink_id)
            key = edge_key_counts.get(edge_pair, 0)
            edge_key_counts[edge_pair] = key + 1

            edges_batch.append((base_operation_id, source_id, sink_id, output_idx, input_idx, key))

    # Compatible mode: deduplicate tensors by address, keep only device tensors
    if not detailed:
        # tensors_batch schema: (tensor_id, shape, dtype, layout, memory_config, device_id, address, buffer_type)
        seen_addresses = {}
        tensor_id_remap = {}
        filtered_tensors = []
        for t in tensors_batch:
            tid, shape, dtype, layout, mem_cfg, dev_id, addr, bt = t
            if dev_id is None:
                continue
            if addr is not None and addr in seen_addresses:
                tensor_id_remap[int(tid) if isinstance(tid, str) else tid] = seen_addresses[addr]
                continue
            kept_id = int(tid) if isinstance(tid, str) else tid
            if addr is not None:
                seen_addresses[addr] = kept_id
            filtered_tensors.append(t)
        tensors_batch = filtered_tensors

        kept_tensor_ids = {int(t[0]) if isinstance(t[0], str) else t[0] for t in tensors_batch}

        def _remap_tid(tid):
            tid_int = int(tid) if isinstance(tid, str) else tid
            return tensor_id_remap.get(tid_int, tid_int)

        input_tensors_batch = [
            (op_id, idx, _remap_tid(tid))
            for op_id, idx, tid in input_tensors_batch
            if _remap_tid(tid) in kept_tensor_ids
        ]
        output_tensors_batch = [
            (op_id, idx, _remap_tid(tid))
            for op_id, idx, tid in output_tensors_batch
            if _remap_tid(tid) in kept_tensor_ids
        ]
        device_tensors_batch = [
            (_remap_tid(tid), dev_id, addr)
            for tid, dev_id, addr in device_tensors_batch
            if _remap_tid(tid) in kept_tensor_ids
        ]
        # Deduplicate device_tensors by (tensor_id, address)
        seen_dt = set()
        filtered_dt = []
        for dt in device_tensors_batch:
            key = (dt[0], dt[2])
            if key not in seen_dt:
                seen_dt.add(key)
                filtered_dt.append(dt)
        device_tensors_batch = filtered_dt

    # Batch inserts
    if captured_graph_batch:
        cursor.executemany("""INSERT INTO captured_graph VALUES (?, ?)""", captured_graph_batch)
    if nodes_batch:
        cursor.executemany("""INSERT INTO nodes VALUES (?, ?, ?, ?)""", nodes_batch)
    if stack_traces_batch:
        cursor.executemany("""INSERT INTO stack_traces VALUES (?, ?)""", stack_traces_batch)
    if operations_batch:
        cursor.executemany("""INSERT OR REPLACE INTO operations VALUES (?, ?, ?)""", operations_batch)
    if operation_arguments_batch:
        cursor.executemany("""INSERT INTO operation_arguments VALUES (?, ?, ?)""", operation_arguments_batch)
    if input_tensors_batch:
        cursor.executemany("""INSERT INTO input_tensors VALUES (?, ?, ?)""", input_tensors_batch)
    if output_tensors_batch:
        cursor.executemany("""INSERT INTO output_tensors VALUES (?, ?, ?)""", output_tensors_batch)
    if tensors_batch:
        cursor.executemany("""INSERT OR IGNORE INTO tensors VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", tensors_batch)
    if device_tensors_batch:
        cursor.executemany("""INSERT INTO device_tensors VALUES (?, ?, ?)""", device_tensors_batch)
    if buffers_batch:
        cursor.executemany("""INSERT INTO buffers VALUES (?, ?, ?, ?, ?, ?)""", buffers_batch)
    if errors_batch:
        cursor.executemany("""INSERT INTO errors VALUES (?, ?, ?, ?)""", errors_batch)
    if edges_batch:
        cursor.executemany("""INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)""", edges_batch)

    # Validate referential integrity
    warnings = _validate_graph_integrity(
        operations_batch,
        tensors_batch,
        input_tensors_batch,
        output_tensors_batch,
        operation_arguments_batch,
        device_tensors_batch,
        buffers_batch,
        devices,
    )
    for w in warnings:
        print(f"WARNING [graph import]: {w}")

    return {
        "operations": len(operations_batch),
        "tensors": len(tensors_batch),
        "device_tensors": len(device_tensors_batch),
        "buffers": len(buffers_batch),
        "nodes": len(nodes_batch),
        "edges": len(edges_batch),
        "stack_traces": len(stack_traces_batch),
        "errors": len(errors_batch),
        "warnings": warnings,
    }


def import_metadata(cursor: sqlite3.Cursor, metadata: dict) -> None:
    """Import report metadata using batch insert."""
    if not metadata:
        return
    batch = [(key, str(value)) for key, value in metadata.items()]
    cursor.executemany("""INSERT OR REPLACE INTO report_metadata VALUES (?, ?)""", batch)


def extract_total_duration_from_graph(graph: list) -> float:
    """Extract total capture duration in seconds."""
    for node in reversed(graph):
        if node.get("node_type") == "capture_end":
            return node.get("duration_ns", 0) / 1e9
    return 0.0


def extract_operation_durations(graph: list) -> dict:
    """Extract per-operation durations as {name: seconds}."""
    durations = {}
    for node in graph:
        if node.get("node_type") == "function_end":
            name = node.get("params", {}).get("name", "unknown")
            durations[name] = node.get("duration_ns", 0) / 1e9
    return durations


def generate_svg(graph: list, output_path: Union[str, Path]) -> Path:
    """
    Generate SVG visualization from graph data.

    Args:
        graph: Graph data (list of nodes from JSON report)
        output_path: Path for output SVG file

    Returns:
        Path to generated SVG file
    """
    # Import here to avoid circular dependency and allow CLI usage without ttnn
    try:
        import ttnn.graph

        output_path = Path(output_path)
        ttnn.graph.visualize(graph, file_name=output_path)
        return output_path
    except ImportError:
        print("Warning: ttnn.graph not available, skipping SVG generation")
        return None


def import_report(
    report_path: Union[str, Path],
    output_dir: Union[str, Path],
    db_name: str = "db.sqlite",
    generate_svgs: bool = False,
    detailed: bool = False,
) -> Path:
    """
    Import a C++ graph capture report into ttnn-visualizer SQLite database.

    This is the main entry point for offline import.

    Args:
        report_path: Path to JSON report file (or directory containing multiple reports)
        output_dir: Directory to create SQLite database in
        db_name: Database filename
        generate_svgs: If True, generate SVG visualizations for each report
        detailed: When False (default), only device tensors are imported and duplicates
            are merged by address to match the legacy Python capture behavior.
            When True, all tensors (host + device) are imported with full detail.

    Returns:
        Path to created database
    """
    report_path = Path(report_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / db_name

    # Create graphs directory if generating SVGs
    graphs_dir = None
    if generate_svgs:
        graphs_dir = output_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        create_database_schema(cursor)

        # Handle single file or directory of reports
        if report_path.is_file():
            report_files = [report_path]
        else:
            report_files = list(report_path.glob("*.json"))

        total_stats = {
            "files": 0,
            "operations": 0,
            "tensors": 0,
            "buffers": 0,
            "edges": 0,
            "devices": 0,
            "errors": 0,
            "stack_traces": 0,
            "svgs": 0,
        }

        for idx, rpath in enumerate(sorted(report_files)):
            with open(rpath, "r") as f:
                report = json.load(f)

            version = report.get("version", 0)
            if version != SUPPORTED_REPORT_VERSION:
                print(f"Warning: {rpath} has version {version}, expected {SUPPORTED_REPORT_VERSION}")
                continue

            devices_data = report.get("devices", [])
            if devices_data:
                device_ids = import_devices(cursor, devices_data)
                total_stats["devices"] += len(device_ids)

            if "graph" in report:
                stats = import_graph(
                    cursor, report["graph"], base_operation_id=idx * 10000, devices=devices_data, detailed=detailed
                )
                total_stats["operations"] += stats["operations"]
                total_stats["tensors"] += stats["tensors"]
                total_stats["device_tensors"] = total_stats.get("device_tensors", 0) + stats.get("device_tensors", 0)
                total_stats["buffers"] += stats["buffers"]
                total_stats["edges"] += stats.get("edges", 0)
                total_stats["errors"] += stats.get("errors", 0)
                total_stats["stack_traces"] += stats.get("stack_traces", 0)

                # Generate SVG if requested
                if generate_svgs and graphs_dir:
                    svg_path = graphs_dir / f"{rpath.stem}.svg"
                    if generate_svg(report["graph"], svg_path):
                        total_stats["svgs"] += 1

            if "metadata" in report:
                import_metadata(cursor, report["metadata"])

            # Save cluster descriptor YAML if present
            if "cluster_descriptor" in report and report["cluster_descriptor"]:
                cluster_path = output_dir / "cluster_descriptor.yaml"
                if not cluster_path.exists():  # Only write once
                    with open(cluster_path, "w") as f:
                        f.write(report["cluster_descriptor"])
                    total_stats["cluster_descriptor"] = True

            # Save mesh coordinate mapping if present (matches old save_mesh_descriptor behavior)
            if "mesh_coordinate_mapping" in report and report["mesh_coordinate_mapping"]:
                mesh_path = output_dir / "mesh_coordinate_mapping.yaml"
                if not mesh_path.exists():  # Only write once
                    with open(mesh_path, "w") as f:
                        f.write(report["mesh_coordinate_mapping"])
                    total_stats["mesh_coordinate_mapping"] = True

            # Import buffer pages if present (when detailed buffer report is enabled)
            if "buffer_pages" in report and report["buffer_pages"]:
                base_op_id = idx * 10000
                buffer_pages_batch = [
                    (
                        base_op_id,
                        page.get("device_id", 0),
                        page.get("address", 0),
                        page.get("core_y", 0),
                        page.get("core_x", 0),
                        page.get("bank_id", 0),
                        page.get("page_index", 0),
                        page.get("page_address", 0),
                        page.get("page_size", 0),
                        page.get("buffer_type", 0),
                    )
                    for page in report["buffer_pages"]
                ]
                cursor.executemany(
                    """INSERT INTO buffer_pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", buffer_pages_batch
                )
                total_stats["buffer_pages"] = total_stats.get("buffer_pages", 0) + len(buffer_pages_batch)

            total_stats["files"] += 1

        conn.commit()
        print(f"Imported {total_stats['files']} report(s) into {db_path}")
        print(f"  - {total_stats['devices']} devices")
        print(f"  - {total_stats['operations']} operations")
        print(f"  - {total_stats['tensors']} tensors")
        if total_stats.get("device_tensors", 0) > 0:
            print(f"  - {total_stats['device_tensors']} device tensor entries")
        print(f"  - {total_stats['buffers']} buffers")
        print(f"  - {total_stats['edges']} edges")
        if total_stats["errors"] > 0:
            print(f"  - {total_stats['errors']} errors captured")
        if total_stats["stack_traces"] > 0:
            print(f"  - {total_stats['stack_traces']} stack traces captured")
        if total_stats.get("buffer_pages", 0) > 0:
            print(f"  - {total_stats['buffer_pages']} buffer pages")
        if total_stats.get("cluster_descriptor"):
            print(f"  - cluster_descriptor.yaml saved")
        if total_stats.get("mesh_coordinate_mapping"):
            print(f"  - mesh_coordinate_mapping.yaml saved")
        if generate_svgs:
            print(f"  - {total_stats['svgs']} SVG visualizations in {graphs_dir}/")

    finally:
        conn.close()

    return db_path


def main():
    """CLI for importing reports."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Import C++ graph capture reports into ttnn-visualizer database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Import single report
    python -m ttnn.graph_report report.json ./visualizer_db/

    # Import all reports from a directory
    python -m ttnn.graph_report ./reports/ ./visualizer_db/

    # Import with SVG visualization generation
    python -m ttnn.graph_report report.json ./visualizer_db/ --svg

    # Then open ttnn-visualizer pointing to ./visualizer_db/
        """,
    )
    parser.add_argument("report_path", help="JSON report file or directory containing reports")
    parser.add_argument("output_dir", help="Directory to create SQLite database in")
    parser.add_argument("--db-name", default="db.sqlite", help="Database filename")
    parser.add_argument("--svg", action="store_true", help="Generate SVG visualizations")

    args = parser.parse_args()
    import_report(args.report_path, args.output_dir, args.db_name, generate_svgs=args.svg)


if __name__ == "__main__":
    main()
