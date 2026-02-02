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
import sqlite3
from pathlib import Path
from typing import Union, Optional, List
import glob

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
            node_counter int,
            operation_name text,
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


def import_graph(cursor: sqlite3.Cursor, graph: list, base_operation_id: int = 0) -> dict:
    """
    Import graph trace into database using batch inserts for performance.

    Extracts and imports:
    - Operations with durations
    - Tensors
    - Buffers
    - Operation arguments
    - Input/output tensor relationships
    - Graph edges

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

    # Store raw graph JSON (single insert is fine)
    cursor.execute(
        """INSERT INTO captured_graph VALUES (?, ?)""",
        (base_operation_id, json.dumps(graph)),
    )

    # Track function start nodes to pair with function end
    function_stack = []
    operation_counter = 0
    tensor_ids_seen = set()

    # First pass: collect all data
    for node in graph:
        node_type = node.get("node_type", "")
        counter = node.get("counter", 0)
        params = node.get("params", {})

        if node_type == "function_start":
            name = params.get("name", "unknown")
            function_stack.append(
                {
                    "counter": counter,
                    "name": name,
                    "arguments": node.get("arguments", []),
                    "input_tensors": node.get("input_tensors", []),
                }
            )

            nodes_batch.append((base_operation_id, counter, operation_counter, name))

            stack_trace = node.get("stack_trace", [])
            if stack_trace:
                stack_traces_batch.append((base_operation_id, counter, name, "\n".join(stack_trace)))

        elif node_type == "function_end":
            name = params.get("name", "unknown")
            duration_ns = node.get("duration_ns", 0)
            duration_s = duration_ns / 1e9 if duration_ns else 0

            operation_id = base_operation_id + operation_counter
            operations_batch.append((operation_id, name, duration_s))

            if function_stack:
                start_node = function_stack.pop()
                for idx, arg in enumerate(start_node.get("arguments", [])):
                    operation_arguments_batch.append((operation_id, f"arg_{idx}", str(arg)))

                for idx, tensor_id in enumerate(start_node.get("input_tensors", [])):
                    input_tensors_batch.append((operation_id, idx, tensor_id))

            # Output tensor relationships from connections
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
                        device_tensors_batch.append((tensor_id, dt.get("device_id"), dt.get("address")))

        elif node_type == "buffer_allocate":
            device_id = int(params.get("device_id", 0))
            address = int(params.get("address", 0))
            size = int(params.get("size", 0))
            buffer_type = 1 if params.get("type") == "L1" else 0
            layout = params.get("layout", "INTERLEAVED")
            layout_int = {"INTERLEAVED": 0, "HEIGHT_SHARDED": 1, "WIDTH_SHARDED": 2, "BLOCK_SHARDED": 3}.get(layout, 0)

            buffers_batch.append((base_operation_id, device_id, address, size, buffer_type, layout_int))

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

    # Batch inserts
    if nodes_batch:
        cursor.executemany("""INSERT INTO nodes VALUES (?, ?, ?, ?)""", nodes_batch)
    if stack_traces_batch:
        cursor.executemany("""INSERT INTO stack_traces VALUES (?, ?, ?, ?)""", stack_traces_batch)
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

    return {
        "operations": len(operations_batch),
        "tensors": len(tensors_batch),
        "device_tensors": len(device_tensors_batch),
        "buffers": len(buffers_batch),
        "nodes": len(nodes_batch),
        "edges": len(edges_batch),
        "stack_traces": len(stack_traces_batch),
        "errors": len(errors_batch),
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
) -> Path:
    """
    Import a C++ graph capture report into ttnn-visualizer SQLite database.

    This is the main entry point for offline import.

    Args:
        report_path: Path to JSON report file (or directory containing multiple reports)
        output_dir: Directory to create SQLite database in
        db_name: Database filename
        generate_svgs: If True, generate SVG visualizations for each report

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

            if "devices" in report:
                device_ids = import_devices(cursor, report["devices"])
                total_stats["devices"] += len(device_ids)

            if "graph" in report:
                stats = import_graph(cursor, report["graph"], base_operation_id=idx * 10000)
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
