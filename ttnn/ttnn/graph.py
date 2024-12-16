# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Callable, Union
from loguru import logger
import pathlib
import graphviz

from ttnn._ttnn.graph import RunMode, begin_graph_capture, end_graph_capture, extract_calltrace


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
