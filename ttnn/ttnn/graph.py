# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
from typing import Callable, Union
from loguru import logger
import pathlib
import graphviz

from ttnn._ttnn.graph import (
    RunMode,
    begin_graph_capture as _cpp_begin_graph_capture,
    end_graph_capture,
    end_graph_capture_to_file as _cpp_end_graph_capture_to_file,
    get_current_report,
    REPORT_VERSION,
    extract_calltrace,
    extract_levelized_graph,
    TensorInfo,
    extract_peak_L1_memory_usage,
    count_intermediate_and_output_tensors,
    extract_output_info,
    extract_output_tensors,
    extract_resource_usage_per_core,
    enable_stack_traces,
    disable_stack_traces,
    is_stack_trace_enabled,
    enable_buffer_pages,
    disable_buffer_pages,
    is_buffer_pages_enabled,
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
    """Wrapper that clears Python I/O state before starting C++ capture."""
    global _python_io_data
    if not is_graph_capture_active():
        _python_io_data = []
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


def end_graph_capture_to_file(report_path):
    """Wrapper that appends Python I/O data to the JSON report."""
    result = _cpp_end_graph_capture_to_file(report_path)
    if _python_io_data:
        _merge_python_io_into_report(report_path)
    return result


def _safe_arg_str(v):
    """Stringify an argument without triggering graph-tracked operations.

    For ttnn.Tensor and torch.Tensor objects, produce a compact summary
    (shape + dtype) instead of calling str() which would read device data
    and pollute the graph capture with spurious operations.
    """
    import ttnn

    if isinstance(v, ttnn.Tensor):
        try:
            return f"ttnn.Tensor(shape={v.shape}, dtype={v.dtype})"
        except Exception:
            return "<ttnn.Tensor>"
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

    _python_io_data.append(
        {
            "name": name,
            "arguments": args_dict,
            "input_tensor_ids": input_tensor_ids,
        }
    )


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


def _merge_python_io_into_report(report_path):
    report_path = pathlib.Path(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)
    report["python_io"] = _python_io_data
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


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
            node_string = format_string.format("Add Tensor: " + node["params"]["tensor_id"])
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
