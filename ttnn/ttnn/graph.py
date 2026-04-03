# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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


def enable_python_stack_traces():
    """Enable capturing Python call stacks in graph trace records."""
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
        path = pathlib.Path(frame.filename).resolve()
        # Match against path relative to root so globs like **/_pytest/** work on absolute paths
        try:
            path_for_match = path.relative_to(path.anchor)
        except ValueError:
            path_for_match = path
        if any(path_for_match.match(p) for p in _STACK_TRACE_INTERNAL_PATTERNS):
            continue
        result.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n    {frame.line}\n')

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


def begin_graph_capture(run_mode=None):
    """Wrapper that clears Python I/O state before starting C++ capture.

    Automatically enables Python I/O recording so that
    ``end_graph_capture_to_file`` can embed operation arguments,
    tensor IDs, and (if enabled) stack traces in the JSON report.

    When graph capture is started from C++ (e.g. ``MemoryUsageTracker``),
    this wrapper is bypassed and Python I/O recording stays disabled,
    avoiding the associated overhead.
    """
    global _python_io_data, _python_io_recording_enabled
    if not is_graph_capture_active():
        _python_io_data = []
        _python_io_recording_enabled = True
        import ttnn

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
    capture session ends.
    """
    global _python_io_recording_enabled
    result = _cpp_end_graph_capture()
    if not is_graph_capture_active():
        _python_io_recording_enabled = False
    return result


def end_graph_capture_to_file(report_path):
    """Wrapper that appends Python I/O data to the JSON report."""
    global _python_io_recording_enabled
    result_str = _cpp_end_graph_capture_to_file(report_path)
    if _python_io_data:
        _write_python_io_sidecar(report_path)
    if not is_graph_capture_active():
        _python_io_recording_enabled = False
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
        record["python_stack_trace"] = _capture_python_stack_trace()

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
