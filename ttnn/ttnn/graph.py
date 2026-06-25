# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import traceback
from typing import Callable, Union
from loguru import logger
import pathlib
import graphviz

from ttnn._ttnn.graph import (
    RunMode,
    begin_graph_capture as _cpp_begin_graph_capture,
    end_graph_capture as _cpp_end_graph_capture,
    end_graph_capture_to_file as _cpp_end_graph_capture_to_file,
    REPORT_VERSION,
    extract_calltrace,
    extract_levelized_graph,
    TensorInfo,
    extract_peak_L1_memory_usage,
    count_intermediate_and_output_tensors,
    extract_output_info,
    extract_output_tensors,
    extract_resource_usage_per_core,
    enable_detailed_buffer_tracing,
    disable_detailed_buffer_tracing,
    is_detailed_buffer_tracing_enabled,
    is_graph_capture_active,
    track_function_start,
    track_function_end,
    # Tier-1 up-front parallel precompile (ttnn/up_front_compile.hpp)
    up_front_begin_collect,
    up_front_end_collect,
    up_front_num_unique,
    up_front_num_collected,
    up_front_clear,
    up_front_compile,
)

from ttnn.graph_report import (
    import_report,
    extract_total_duration_from_graph,
    extract_operation_durations,
)


# ---------------------------------------------------------------------------
# Python-level I/O tracking
# ---------------------------------------------------------------------------
# The C++ graph trace does not capture Python function arguments or return
# values.  The decorator records them here so that end_graph_capture_to_file
# can embed them in the JSON report.  The offline importer then uses them
# to set correct input_tensors / output_tensors associations.

_python_io_data: list = []
_python_io_recording_enabled: bool = False
_python_stack_traces_enabled: bool = False
_python_stack_traces_auto_for_session: bool = False
_comparison_records_data: dict = {}

COMPARISON_RECORDS_SIDECAR_SUFFIX = ".comparison_records.json"


def _new_comparison_records_data() -> dict:
    return {
        "version": 1,
        "local_tensor_comparison_records": [],
        "global_tensor_comparison_records": [],
        "tensors": [],
    }


def reset_comparison_records_data():
    """Clear accumulated comparison-mode sidecar data (call at test boundaries)."""
    global _comparison_records_data
    _comparison_records_data = _new_comparison_records_data()


# Glob patterns for frames to strip from stack traces (pathlib-style).
# Matches ttnn internals (decorators/graph), pytest, pluggy, and the pytest entry script.
_STACK_TRACE_INTERNAL_PATTERNS = (
    "**/ttnn/**/decorators.py",
    "**/ttnn/decorators.py",
    "**/ttnn/**/graph.py",
    "**/ttnn/graph.py",
    "**/_pytest/**",
    "**/_pytest/config/__init__.py",
    "**/pluggy/**",
    "**/bin/pytest",
)


def enable_python_io_recording():
    """Enable recording Python-level operation arguments and tensor IDs during graph capture.

    When enabled, each operation records its kwargs, tensor summaries, and
    (optionally) Python stack traces.  This is required for
    ``end_graph_capture_to_file`` / ``graph_report`` but adds significant
    overhead — disable it when graph capture is used only for memory tracking.

    Automatically enabled by :func:`begin_graph_capture` and disabled by
    :func:`end_graph_capture` / :func:`end_graph_capture_to_file`.
    """
    global _python_io_recording_enabled
    _python_io_recording_enabled = True


def disable_python_io_recording():
    """Disable Python-level I/O recording during graph capture."""
    global _python_io_recording_enabled
    _python_io_recording_enabled = False


def is_python_io_recording_enabled() -> bool:
    """Return whether Python I/O recording is currently enabled."""
    return _python_io_recording_enabled


def _configure_python_stack_traces_for_outer_graph_capture(ttnn_mod) -> None:
    """Outermost ``begin_graph_capture`` only: turn on Python stacks if ``CONFIG`` allows.

    Stack traces are enabled when either ``enable_graph_python_stack_traces`` or
    ``enable_detailed_tensor_report`` on ``ttnn_mod.CONFIG`` is true (each defaults to
    false if missing on older ``_ttnn`` builds).  The latter is required for
    ``tensor_lifetime`` source file/line columns in :mod:`ttnn.graph_report`.
    Sets ``_python_stack_traces_auto_for_session`` so the matching
    ``end_graph_capture`` / ``end_graph_capture_to_file`` can disable traces again
    when this path enabled them.
    """
    global _python_stack_traces_auto_for_session
    if _python_stack_traces_enabled:
        _python_stack_traces_auto_for_session = False
        return
    enable_traces = getattr(ttnn_mod.CONFIG, "enable_graph_python_stack_traces", False) or getattr(
        ttnn_mod.CONFIG, "enable_detailed_tensor_report", False
    )
    if enable_traces:
        enable_python_stack_traces()
        _python_stack_traces_auto_for_session = True
    else:
        _python_stack_traces_auto_for_session = False


def enable_python_stack_traces():
    """Enable capturing Python call stacks in graph trace records.

    Ignores ``ttnn.CONFIG.enable_graph_python_stack_traces`` (use for tests, for
    :func:`full_graph_capture`, or explicit ``record_python_operation`` use
    outside graph capture).
    """
    global _python_stack_traces_enabled
    _python_stack_traces_enabled = True


def disable_python_stack_traces():
    """Disable capturing Python call stacks in graph trace records."""
    global _python_stack_traces_enabled
    _python_stack_traces_enabled = False


def is_python_stack_trace_enabled() -> bool:
    """Return whether Python stack trace capture is currently enabled."""
    return _python_stack_traces_enabled


def _capture_python_stack_trace() -> list[str]:
    """Capture the current Python call stack, filtering out ttnn internals.

    Returns a list of formatted frame strings (file:line in function) with
    infrastructure frames from decorators.py and graph.py removed.
    """
    frames = traceback.extract_stack()
    result = []

    for frame in frames:
        try:
            filename = str(frame.filename)
            path = pathlib.PurePath(filename)

            # Match without filesystem resolution to avoid fragile behavior during capture
            if any(path.match(p) for p in _STACK_TRACE_INTERNAL_PATTERNS):
                continue

            result.append(f'  File "{filename}", line {frame.lineno}, in {frame.name}\n    {frame.line}\n')
        except Exception:
            # Never let stack-trace capture break graph capture
            continue

    return result[::-1]


def _collect_tensor_ids(value) -> list:
    """Recursively extract tensor_id ints from ttnn.Tensor and torch.Tensor objects."""
    import ttnn

    try:
        import torch

        _tensor_types = (ttnn.Tensor, torch.Tensor)
    except ImportError:
        _tensor_types = (ttnn.Tensor,)

    ids: list[int] = []
    if isinstance(value, _tensor_types):
        tid = getattr(value, "tensor_id", None)
        if tid is not None:
            ids.append(int(tid))
    elif isinstance(value, (list, tuple)):
        for item in value:
            ids.extend(_collect_tensor_ids(item))
    return ids


def begin_graph_capture(run_mode=None, *, _internal=False):
    """Wrapper that clears Python I/O state before starting C++ capture.

    Automatically enables Python I/O recording so that
    ``end_graph_capture_to_file`` can embed operation arguments,
    tensor IDs, and (when enabled) Python stack traces in the JSON report / sidecar.

    On the outermost Python-started capture only, Python stack traces may be
    turned on by ``_configure_python_stack_traces_for_outer_graph_capture`` when
    ``ttnn.CONFIG.enable_graph_python_stack_traces`` or
    ``ttnn.CONFIG.enable_detailed_tensor_report`` is true (set via
    ``TTNN_CONFIG_OVERRIDES``).  Detailed tensor reporting enables traces for
    ``tensor_lifetime`` source file/line columns in :mod:`ttnn.graph_report`.

    If traces were already enabled before that configure step (for example
    after :func:`enable_python_stack_traces` or from :func:`full_graph_capture`),
    the outer session does not auto-disable them on end.

    When the outermost session ends, :func:`end_graph_capture` /
    :func:`end_graph_capture_to_file` turn traces off again only if this
    outer begin turned them on (internal ``_python_stack_traces_auto_for_session``).

    When graph capture is started from C++ (e.g. ``MemoryUsageTracker``),
    this wrapper is bypassed and Python I/O recording stays disabled,
    avoiding the associated overhead.
    """
    global _python_io_data
    global _python_io_recording_enabled
    global _python_stack_traces_auto_for_session
    global _comparison_records_data
    if not is_graph_capture_active():
        import ttnn

        _python_io_data = []
        # A user-initiated (non-internal) outermost capture starts a fresh comparison
        # sidecar scope: any records left over from earlier comparison-mode activity
        # in this process must not leak into this capture's sidecar. The per-op
        # auto-capture inside the operation wrapper passes _internal=True so that
        # comparison records still accumulate across per-op sessions (comparison mode
        # without enable_graph_report, later drained by flush_comparison_records_to_db).
        if not _internal:
            _comparison_records_data = _new_comparison_records_data()
        elif not _comparison_records_data:
            _comparison_records_data = _new_comparison_records_data()
        _python_io_recording_enabled = True
        _configure_python_stack_traces_for_outer_graph_capture(ttnn)

        if ttnn.CONFIG.enable_fast_runtime_mode:
            logger.warning(
                "Graph capture started with enable_fast_runtime_mode=true (fast dispatch). "
                "FastOperation records arguments, tensor IDs, and output tensor IDs, "
                "but does not produce per-operation captured sub-graphs. "
                "For full captured-graph detail, disable fast runtime mode via "
                "TTNN_CONFIG_OVERRIDES or ttnn.manage_config."
            )
    if run_mode is None:
        return _cpp_begin_graph_capture()
    return _cpp_begin_graph_capture(run_mode)


def end_graph_capture():
    """End graph capture and return the captured graph.

    Automatically disables Python I/O recording when the outermost
    capture session ends.  If Python stack traces were enabled by auto-capture they are disabled again.
    """
    global _python_io_recording_enabled, _python_stack_traces_auto_for_session
    result = _cpp_end_graph_capture()
    if not is_graph_capture_active():
        _python_io_recording_enabled = False
        if _python_stack_traces_auto_for_session:
            disable_python_stack_traces()
            _python_stack_traces_auto_for_session = False
    return result


def end_graph_capture_to_file(report_path):
    """Wrapper that appends Python I/O data to the JSON report."""
    global _python_io_recording_enabled, _python_stack_traces_auto_for_session
    result_str = _cpp_end_graph_capture_to_file(report_path)
    if _python_io_data:
        _write_python_io_sidecar(report_path)
    if has_comparison_records():
        _write_comparison_records_sidecar(report_path)
        reset_comparison_records_data()
    if not is_graph_capture_active():
        _python_io_recording_enabled = False
        if _python_stack_traces_auto_for_session:
            disable_python_stack_traces()
            _python_stack_traces_auto_for_session = False
    return json.loads(result_str)


def _ttnn_tensor_summary(t) -> str:
    """Build a rich one-line summary of a ttnn.Tensor.

    All properties accessed here are always present on ttnn.Tensor and never
    trigger device data reads:
      - shape, dtype, layout, tensor_id: plain attributes on every tensor
      - memory_config(): stored in TensorSpec, available for host and device tensors
      - storage_type(), is_allocated(): always safe
      - device(): returns None for host tensors, MeshDevice* otherwise
      - tensor_topology(): always returns a TensorTopology reference
    """
    info: dict = {}

    info["shape"] = t.shape
    info["dtype"] = t.dtype
    info["layout"] = t.layout
    info["memory_config"] = t.memory_config()
    info["storage_type"] = t.storage_type()
    info["tensor_id"] = t.tensor_id
    info["is_allocated"] = t.is_allocated()

    device = t.device()
    if device is not None:
        info["device_id"] = device.id()
        info["mesh_shape"] = str(device.shape)

    topology = t.tensor_topology()
    mesh_coords = topology.mesh_coords()
    if mesh_coords:
        info["mesh_coords"] = [str(c) for c in mesh_coords]
    dist_shape = topology.distribution_shape()
    if dist_shape is not None:
        info["distribution_shape"] = str(dist_shape)
    placements = topology.placements()
    if placements:
        info["placements"] = [str(p) for p in placements]

    parts = [f"{k}={v}" for k, v in info.items() if v is not None]
    return f"ttnn.Tensor({', '.join(parts)})"


def _safe_arg_str(v):
    """Stringify a function argument without triggering graph-tracked operations."""
    import ttnn

    if isinstance(v, ttnn.Tensor):
        return _ttnn_tensor_summary(v)
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return f"torch.Tensor(shape={list(v.shape)}, dtype={v.dtype})"
    except ImportError:
        pass
    return str(v)


def record_python_operation(name, function_args, function_kwargs):
    """Record a Python-level operation's arguments and I/O tensor ids.

    Called from ``FastOperation.__call__`` and ``runtime_decorator.call_wrapper``
    to capture the Python-visible arguments (named kwargs + positional args)
    that the C++ graph trace does not see.
    """
    args_dict = {}
    for k, v in function_kwargs.items():
        args_dict[k] = _safe_arg_str(v)
    for idx, v in enumerate(function_args):
        args_dict[str(idx)] = _safe_arg_str(v)

    input_tensor_ids = _collect_tensor_ids((*function_args, *function_kwargs.values()))

    record = {
        "name": name,
        "arguments": args_dict,
        "input_tensor_ids": input_tensor_ids,
    }

    if _python_stack_traces_enabled:
        try:
            record["python_stack_trace"] = _capture_python_stack_trace()
        except Exception:
            record["python_stack_trace"] = []

    _python_io_data.append(record)


def store_output_tensor_ids(output_tensor_ids):
    """Attach output tensor IDs to the most recent _python_io_data entry.

    Called from both ``FastOperation.__call__`` and ``runtime_decorator.call_wrapper``.
    """
    if _python_io_data:
        _python_io_data[-1]["output_tensor_ids"] = output_tensor_ids


def store_captured_graph(captured_graph_json):
    """Attach a per-op captured graph to the most recent _python_io_data entry.

    Called from ``runtime_decorator.call_wrapper`` (slow dispatch) right after
    ``end_graph_capture()``.  ``[-1]`` is always the correct entry since
    ``record_python_operation`` is called first for the same operation.
    """
    if _python_io_data:
        _python_io_data[-1]["captured_graph"] = captured_graph_json


def record_tensor_comparison_data(
    *,
    local_tensor_comparison_records=None,
    global_tensor_comparison_records=None,
    tensors=None,
):
    """Record comparison-mode sidecar data for offline graph_report import."""
    global _comparison_records_data
    if not _comparison_records_data:
        _comparison_records_data = _new_comparison_records_data()

    if local_tensor_comparison_records:
        _comparison_records_data["local_tensor_comparison_records"].extend(local_tensor_comparison_records)
    if global_tensor_comparison_records:
        _comparison_records_data["global_tensor_comparison_records"].extend(global_tensor_comparison_records)
    if tensors:
        existing_tensor_ids = {tensor["tensor_id"] for tensor in _comparison_records_data["tensors"]}
        for tensor in tensors:
            if tensor["tensor_id"] in existing_tensor_ids:
                continue
            _comparison_records_data["tensors"].append(tensor)
            existing_tensor_ids.add(tensor["tensor_id"])


def has_comparison_records():
    return bool(
        _comparison_records_data
        and (
            _comparison_records_data["local_tensor_comparison_records"]
            or _comparison_records_data["global_tensor_comparison_records"]
            or _comparison_records_data["tensors"]
        )
    )


def _write_python_io_sidecar(report_path):
    """Write python_io data as a sidecar JSON file next to the main report.

    Avoids the expensive read-modify-write cycle on potentially huge
    (hundreds of MB) graph capture files.  The importer picks up the
    sidecar automatically when it sits alongside the main report.
    """
    report_path = pathlib.Path(report_path)
    sidecar_path = report_path.with_suffix(".python_io.json")
    with open(sidecar_path, "w") as f:
        json.dump(_python_io_data, f)


def _write_comparison_records_sidecar(report_path):
    """Write comparison-mode data next to the main graph capture report."""
    report_path = pathlib.Path(report_path)
    sidecar_path = report_path.with_suffix(COMPARISON_RECORDS_SIDECAR_SUFFIX)
    with open(sidecar_path, "w") as f:
        json.dump(_comparison_records_data, f)


def flush_comparison_records_to_db(report_dir):
    """Write accumulated comparison-mode records into report_dir/db.sqlite.

    Used when enable_comparison_mode is on without a full graph report import
    (enable_graph_report=false). PCC still runs during the model; this persists
    local/global_tensor_comparison_records for the visualizer.
    """
    global _comparison_records_data

    if not has_comparison_records():
        return

    import sqlite3

    from ttnn.graph_report import (
        COMPARISON_RECORDS_FALLBACK_NAME,
        create_database_schema,
        import_tensor_comparison_records,
        save_database_schema_version,
    )

    report_dir = pathlib.Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = report_dir / COMPARISON_RECORDS_FALLBACK_NAME
    with open(sidecar_path, "w") as f:
        json.dump(_comparison_records_data, f)

    rank = 0
    try:
        from ttnn._ttnn.multi_device import get_rank, is_initialized

        if is_initialized():
            rank = int(get_rank())
    except (ImportError, OSError):
        pass

    conn = sqlite3.connect(report_dir / "db.sqlite")
    cursor = conn.cursor()
    try:
        create_database_schema(cursor)
        save_database_schema_version(cursor)
        import_tensor_comparison_records(cursor, _comparison_records_data, rank=rank)
        conn.commit()
    finally:
        conn.close()
        reset_comparison_records_data()


@contextlib.contextmanager
def full_graph_capture(report_path: str, *, run_mode=None, slow_dispatch: bool = True):
    """Context manager that captures a complete graph trace with all features enabled.

    Enables Python I/O recording, Python stack traces, and detailed buffer
    tracing.  Optionally switches to slow dispatch (``Operation`` decorator)
    for per-operation captured sub-graphs.

    On exit the trace is written to *report_path* and all settings are
    restored to their previous values.

    Example::

        with ttnn.graph.full_graph_capture("my_report.json"):
            # ... run your model ...

    Args:
        report_path: Destination JSON file for the graph report.
        run_mode: ``RunMode.NORMAL`` (default) or ``RunMode.NO_DISPATCH``.
        slow_dispatch: If ``True`` (default), temporarily disables
            ``enable_fast_runtime_mode`` so ``Operation`` produces
            per-operation captured sub-graphs.
    """
    import ttnn

    prev_stack_traces = _python_stack_traces_enabled
    prev_buffer_tracing = is_detailed_buffer_tracing_enabled()
    prev_fast_runtime = ttnn.CONFIG.enable_fast_runtime_mode

    enable_python_stack_traces()
    enable_detailed_buffer_tracing()
    if slow_dispatch:
        ttnn.CONFIG.enable_fast_runtime_mode = False

    try:
        begin_graph_capture(run_mode if run_mode is not None else RunMode.NORMAL)
        yield
        end_graph_capture_to_file(report_path)
    finally:
        if is_graph_capture_active():
            try:
                _cpp_end_graph_capture()
            except Exception:
                pass
        if not prev_buffer_tracing:
            disable_detailed_buffer_tracing()
        if not prev_stack_traces:
            disable_python_stack_traces()
        ttnn.CONFIG.enable_fast_runtime_mode = prev_fast_runtime


class ExitStackWithPop(contextlib.ExitStack):
    def pop(self):
        _, context_manager = self._exit_callbacks.pop()
        context_manager(None, None, None)


def pretty_format(captured_graph):
    output = ""
    tabs = ""
    for node in captured_graph:
        if node["node_type"] == "function_end":
            tabs = tabs[:-4]

        length = 50 - len(tabs)
        format_string = f"{{:{length}}}"

        if node["node_type"] == "capture_start":
            node_string = format_string.format("Capture Start")
        elif node["node_type"] == "capture_end":
            node_string = format_string.format("Capture End")
        elif node["node_type"] == "buffer":
            node_string = format_string.format("Add Device Buffer")
        elif node["node_type"] == "buffer_allocate":
            node_string = format_string.format("Allocate Device Buffer")
        elif node["node_type"] == "buffer_deallocate":
            node_string = format_string.format("Deallocate Device Buffer")
        elif node["node_type"] == "function_start":
            node_string = format_string.format("Function Start: " + node["params"]["name"])
        elif node["node_type"] == "function_end":
            node_string = format_string.format("Function End:   " + node["params"]["name"])
        elif node["node_type"] == "tensor":
            node_string = format_string.format("Add Tensor: " + str(node["params"]["tensor_id"]))
        elif node["node_type"] == "circular_buffer_allocate":
            node_string = format_string.format("Allocate Circular Buffer")
        elif node["node_type"] == "circular_buffer_deallocate_all":
            node_string = format_string.format("Deallocate All Circular Buffers")
        else:
            raise ValueError(f"Unknown node type: {node['node_type']}")

        output += f"{tabs}{node_string}\n"

        if node["node_type"] == "function_start":
            tabs += "    "

    return output


def pretty_print(captured_graph):
    print(pretty_format(captured_graph))


def visualize_node(
    graphviz_graph,
    node,
):
    node_type = node["node_type"]
    params = node["params"]

    label = node_type
    if node_type == "function_start":
        function_name = params["name"]
        label += f"\n{function_name}"
    elif node_type == "function_end":
        function_name = params["name"]
        label += f"\n{function_name}"
    elif node_type == "tensor":
        tensor_id = params["tensor_id"]
        label += f"\n{tensor_id}"

    node_id = node["counter"]
    graphviz_graph.node(
        f"node_{node_id}",
        label,
        fillcolor="#DCDCDC",
    )


def visualize_edge(graphviz_graph, source_node_id, sink_node_id):
    graphviz_graph.edge(f"node_{source_node_id}", f"node_{sink_node_id}")


_LEVEL_COLORS = [
    "#0000ff80",
    "#ee00ee80   ",
    "#ff000080",
    "#eeee0080",
    "#00ff0080",
    "#00eeee80",
]


def _visualize(
    context_stack,
    graphviz_graph,
    captured_graph,
    *,
    file_name,
    visualize_node,
    visualize_edge,
) -> graphviz.Digraph:
    subgraph_counter = 0
    graph_stack = []
    level = 0
    for node in captured_graph:
        if node["node_type"] == "function_start":
            graph_stack.append(graphviz_graph)
            graphviz_graph = context_stack.enter_context(graphviz_graph.subgraph(name=f"subgraph_{subgraph_counter}"))
            subgraph_counter += 1
            level += 1

            graphviz_graph.attr(
                fontcolor="black",
                bgcolor=_LEVEL_COLORS[level % len(_LEVEL_COLORS)],
                cluster="true",
                label=node["params"]["name"],
                rankdir="TB",
                shape="hexagon",
            )
            graphviz_graph.node_attr["style"] = "filled"

        visualize_node(graphviz_graph, node)

        if node["node_type"] == "function_end":
            context_stack.pop()
            graphviz_graph = graph_stack.pop()
            level -= 1

    if graph_stack:
        raise ValueError("Call stack not empty")

    for node in captured_graph:
        for connection in node["connections"]:
            visualize_edge(graphviz_graph, node["counter"], connection)

    if file_name is not None:
        file_name = pathlib.Path(file_name)
        if file_name.suffix not in {".svg", ".png", ".pdf"}:
            raise ValueError(f"file_name must have a .svg, .png or .pdf suffix, not {file_name.suffix}")
        format = file_name.suffix[1:]
        graphviz_graph.render(file_name.with_suffix(""), format=format)
        logger.info(f'Graph visualization saved to "{file_name}"')

    return graphviz_graph


def visualize(
    captured_graph,
    *,
    file_name: Union[pathlib.Path, str] = None,
    visualize_node: Callable = visualize_node,
    visualize_edge: Callable = visualize_edge,
) -> graphviz.Digraph:
    if isinstance(file_name, str):
        file_name = pathlib.Path(file_name)

    graph_attr = {"ordering": "in", "rankdir": "TB"}

    node_attr = {
        "style": "filled",
        "border": "1",
        "fontsize": "10",
        "ranksep": "0.1",
        "height": "0.2",
        "fontname": "Linux libertine",
        "margin": "0",
        "shape": "box",
    }

    edge_attr = {
        "fontsize": "10",
    }

    graphviz_graph = graphviz.Digraph(
        engine="dot",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )

    with ExitStackWithPop() as context_stack:
        return _visualize(
            context_stack,
            graphviz_graph,
            captured_graph,
            file_name=file_name,
            visualize_node=visualize_node,
            visualize_edge=visualize_edge,
        )
