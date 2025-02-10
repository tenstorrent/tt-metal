# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import io
import math
import shutil
from typing import Any

from loguru import logger
import networkx as nx

logger.disable("ttnn.torch_tracer")

import ttnn.torch_tracer

import ttnn

TracedTensor = ttnn.torch_tracer.TracedTensor
TracedTorchTensor = ttnn.torch_tracer.TracedTorchTensor

TorchTensor = ttnn.torch_tracer.TorchTensor
TorchParameter = ttnn.torch_tracer.TorchParameter
TorchFunction = ttnn.torch_tracer.TorchFunction
TorchModule = ttnn.torch_tracer.TorchModule
TorchModuleInput = ttnn.torch_tracer.TorchModuleInput

PositionalArgumentName = ttnn.torch_tracer.PositionalArgumentName
InputTensorIndex = ttnn.torch_tracer.InputTensorIndex


duration_to_string = ttnn.torch_tracer.duration_to_string
get_input_tensors = ttnn.torch_tracer.get_input_tensors
get_arg_name_value_pairs = ttnn.torch_tracer.get_arg_name_value_pairs

LEVEL_COLORS = ttnn.torch_tracer.LEVEL_COLORS

_visualize = ttnn.torch_tracer._visualize


@dataclasses.dataclass
class TTNNTensor:
    def to_string(self, verbose: bool = False) -> str:
        return "ttnn.Tensor"

    __repr__ = to_string


@dataclasses.dataclass
class TTNNOperation:
    pretty_name: str
    operation: Any
    arg_name_value_pairs: Any

    def to_string(self, verbose: bool = False) -> str:
        return self.pretty_name

    __repr__ = to_string


class TracedTTNNTensor(ttnn.Tensor, ttnn.torch_tracer.TracedTensor):
    def __init__(self, tensor: ttnn.Tensor, *, graph: nx.MultiDiGraph, node: ttnn.torch_tracer.Node, output_index: int):
        super().__init__(tensor)
        self.graph: nx.MultiDiGraph = graph
        self.node: ttnn.torch_tracer.Node = node
        self.output_index: int = output_index

    @property
    def name(self) -> str:
        return self.node.name


def create_ttnn_input_tensor(tensor: ttnn.Tensor) -> TracedTTNNTensor:
    unique_id = ttnn.torch_tracer.get_unique_id()
    node_name = f"ttnn_input_{unique_id}"
    node = ttnn.torch_tracer.Node(name=node_name, unique_id=unique_id)
    graph = ttnn.torch_tracer.GRAPH_STACK[-1]
    memory_config = ttnn.get_memory_config(tensor)
    graph.add_node(
        node,
        operation=TTNNTensor(),
        shapes=(tensor.shape,),
        dtypes=(tensor.dtype,),
        layouts=(tensor.layout,),
        memory_configs=(memory_config,),
    )
    return TracedTTNNTensor(tensor, graph=graph, node=node, output_index=0)


def preprocess_args_and_kwargs(*function_args, **function_kwargs) -> Any:
    import torch

    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, ttnn.torch_tracer.TracedTensor):
            return arg
        elif isinstance(arg, ttnn.Tensor):
            return create_ttnn_input_tensor(arg)
        elif isinstance(arg, torch.Tensor):
            return ttnn.torch_tracer.create_input_tensor(arg)
        elif isinstance(arg, (tuple, list)):
            return type(arg)([preprocess_arg(element) for element in arg])
        elif isinstance(arg, dict):
            return {key: preprocess_arg(value) for key, value in arg.items()}
        else:
            return arg

    function_args = [preprocess_arg(arg) for arg in function_args]
    function_kwargs = {name: preprocess_arg(arg) for name, arg in function_kwargs.items()}
    return function_args, function_kwargs


def preprocess_return_value(return_value):
    import torch

    output_tensors = []
    if isinstance(return_value, torch.Tensor) and not isinstance(return_value, ttnn.torch_tracer.TracedTorchTensor):
        output_tensors.append(return_value)
    elif isinstance(return_value, ttnn.Tensor) and not isinstance(return_value, TracedTTNNTensor):
        output_tensors.append(create_ttnn_input_tensor(return_value))
    elif isinstance(return_value, ttnn.torch_tracer.TracedTensor):
        output_tensors.append(return_value)
    elif isinstance(return_value, (tuple, list)):
        for value in return_value:
            output_tensors += preprocess_return_value(value)
    elif isinstance(return_value, dict):
        for value in return_value.values():
            output_tensors += preprocess_return_value(value)
    elif return_value is None:
        pass
    else:
        raise ValueError(f"Unexpected type {type(return_value)}")
    return output_tensors


def postprocess_return_value(return_value, output_tensors):
    import torch

    if isinstance(return_value, (torch.Tensor, ttnn.torch_tracer.TracedTorchTensor, ttnn.Tensor, TracedTTNNTensor)):
        output_tensor, *_ = output_tensors
        output_tensors.pop(0)
        return output_tensor
    elif isinstance(return_value, tuple):
        return tuple(postprocess_return_value(value, output_tensors) for value in return_value)
    elif isinstance(return_value, dict):
        return {name: postprocess_return_value(value, output_tensors) for name, value in return_value.items()}
    else:
        return return_value


def trace_ttnn_operation(pretty_operation_name, operation):
    import torch

    def call_wrapper(*function_args, **function_kwargs):
        operation_id = ttnn._ttnn.get_python_operation_id()

        original_function_args = function_args
        original_function_kwargs = function_kwargs
        function_args, function_kwargs = preprocess_args_and_kwargs(*function_args, **function_kwargs)
        input_tensors = get_input_tensors(function_args) + get_input_tensors(function_kwargs)

        GRAPH_STACK.append(nx.MultiDiGraph())

        node_name = f"{pretty_operation_name}_{ttnn.torch_tracer.get_unique_id()}"

        operation_return_type = operation(*function_args, **function_kwargs)

        output_tensors = preprocess_return_value(operation_return_type)

        GRAPH_STACK.pop()

        shapes = tuple(tensor.shape for tensor in output_tensors)
        dtypes = tuple(tensor.dtype for tensor in output_tensors)
        layouts = tuple(tensor.layout for tensor in output_tensors)
        memory_configs = tuple(
            ttnn.get_memory_config(tensor) if isinstance(tensor, ttnn.Tensor) else None for tensor in output_tensors
        )

        unique_id = ttnn.torch_tracer.get_unique_id()
        node_name = f"{pretty_operation_name}_{unique_id}"
        node = ttnn.torch_tracer.Node(name=node_name, unique_id=unique_id)

        graph = ttnn.torch_tracer.GRAPH_STACK[-1]
        for tensor in input_tensors:
            if tensor.graph is not graph:
                graph = nx.compose(graph, tensor.graph)
                ttnn.torch_tracer.GRAPH_STACK[-1] = graph

        try:
            arg_name_value_pairs = get_arg_name_value_pairs(
                operation, function_args=original_function_args, function_kwargs=original_function_kwargs
            )
        except Exception as e:
            arg_name_value_pairs = []

        graph.add_node(
            node,
            operation=TTNNOperation(
                pretty_name=pretty_operation_name, operation=operation, arg_name_value_pairs=arg_name_value_pairs
            ),
            shapes=shapes,
            dtypes=dtypes,
            layouts=layouts,
            memory_configs=memory_configs,
            operation_id=operation_id,
        )
        for input_index, tensor in enumerate(input_tensors):
            graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )

        output_tensors = [
            (
                ttnn.torch_tracer.TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
                if isinstance(tensor, torch.Tensor)
                else TracedTTNNTensor(tensor, graph=graph, node=node, output_index=output_index)
            )
            for output_index, tensor in enumerate(output_tensors)
        ]
        return postprocess_return_value(operation_return_type, output_tensors)

    return call_wrapper


def layout_to_string(layout):
    if layout is None:
        return ""

    if not isinstance(layout, ttnn.Layout):
        return layout

    if layout == ttnn.Layout.ROW_MAJOR:
        return "Layout: ROW_MAJOR"
    elif layout == ttnn.Layout.TILE:
        return "Layout: TILE"
    else:
        raise ValueError(f"Unknown layout: {layout}")


def memory_config_to_string(memory_config):
    if memory_config is None:
        return ""

    string = f"Memory: "
    if memory_config.buffer_type == ttnn.BufferType.DRAM:
        string += "DRAM"
    elif memory_config.buffer_type == ttnn.BufferType.L1:
        string += "L1"
    else:
        raise ValueError(f"Unknown buffer type: {memory_config.buffer_type}")

    if memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        string += ", INTERLEAVED"
    elif memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        string += ", HEIGHT_SHARDED"
    elif memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        string += ", WIDTH_SHARDED"
    elif memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        string += ", BLOCK_SHARDED"
    else:
        raise ValueError(f"Unknown tensor memory layout: {memory_config.memory_layout}")

    if memory_config.shard_spec is not None:
        string += "\n"
        string += f"Shard Grid: {memory_config.shard_spec.grid}\n"
        string += f"Shard Shape: {memory_config.shard_spec.shape}\n"
        string += f"Shard Orientation: "
        if memory_config.shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR:
            string += "ROW_MAJOR"
        elif memory_config.shard_spec.orientation == ttnn.ShardOrientation.COL_MAJOR:
            string += "COL_MAJOR"
        else:
            raise ValueError(f"Unknown shard orientation: {memory_config.shard_spec.orientation}")

    return string


def visualize_node(
    graphviz_graph,
    graph,
    node,
    max_depth,
    visualize_node,
    visualize_edge,
    level,
    verbose,
):
    attributes = graph.nodes[node]
    operation = attributes["operation"]
    operation_id = attributes.get("operation_id", None)

    input_tensors = []
    input_layouts = []
    input_memory_configs = []
    for source_node, _, edge_data in graph.in_edges(node, data=True):
        input_attributes = graph.nodes[source_node]
        source_output_index = edge_data["source_output_index"]
        input_shape = input_attributes["shapes"][source_output_index]
        input_dtype = input_attributes["dtypes"][source_output_index]
        input_tensors.append((input_shape, input_dtype))
        if "layouts" in input_attributes:
            input_layouts.append(input_attributes["layouts"][source_output_index])
        else:
            input_layouts.append(None)
        if "memory_configs" in input_attributes:
            input_memory_configs.append(input_attributes["memory_configs"][source_output_index])
        else:
            input_memory_configs.append(None)

    output_shapes = attributes["shapes"]
    output_dtypes = attributes["dtypes"]
    output_tensors = tuple((shape, dtype) for shape, dtype in zip(output_shapes, output_dtypes))

    output_layouts = []
    output_memory_configs = []
    for output_index in range(len(output_tensors)):
        if "layouts" in attributes:
            output_layouts.append(attributes["layouts"][output_index])
        else:
            output_layouts.append(None)
        if "memory_configs" in attributes:
            output_memory_configs.append(attributes["memory_configs"][output_index])
        else:
            output_memory_configs.append(None)

    label = operation.to_string(verbose=verbose)
    if operation_id is not None:
        label = f"{attributes['operation_id']}: {label}"

    duration = attributes.get("duration", None)
    if duration is not None:
        label = f"{label}\nduration: {duration_to_string(duration)}"

    num_columns = max(len(input_tensors), len(output_tensors))

    table_lable = label.replace("\n", "<BR/>")
    table = f"""<
            <TABLE BORDER="{0}" CELLBORDER="{1}"
            CELLSPACING="{1}" CELLPADDING="{1}">"""

    def compute_even_column_sizes(num_columns, num_tensors):
        if num_tensors == 0:
            return []
        column_size = math.ceil(num_columns // num_tensors)
        column_sizes = []
        remaining = num_columns
        for _ in range(num_tensors):
            if remaining > column_size:
                column_sizes.append(column_size)
            else:
                column_sizes.append(remaining)
            remaining -= column_size
        return column_sizes

    rowspan = 2 if input_tensors and output_tensors else 1

    table += f"""
            <TR>
                <TD ROWSPAN="{rowspan}">{table_lable}</TD>
            """
    if input_tensors:
        input_column_sizes = compute_even_column_sizes(num_columns, len(input_tensors))
        for index, (shape, dtype) in enumerate(input_tensors):
            layout = layout_to_string(input_layouts[index])
            memory_config = memory_config_to_string(input_memory_configs[index])
            memory_config = memory_config.replace("\n", "<BR/>")
            column_size = input_column_sizes[index]
            table = (
                table
                + f"""
                    <TD PORT="${index}" COLSPAN="{column_size}">Input {index}<BR/>{shape}<BR/>{dtype}<BR/>{layout}<BR/>{memory_config}</TD>
                """
            )
    table += "</TR>"

    if output_tensors:
        table += "<TR>"
        output_column_sizes = compute_even_column_sizes(num_columns, len(output_tensors))
        for index, (shape, dtype) in enumerate(output_tensors):
            layout = layout_to_string(output_layouts[index])
            memory_config = memory_config_to_string(output_memory_configs[index])
            memory_config = memory_config.replace("\n", "<BR/>")
            column_size = output_column_sizes[index]
            table += f"""
                    <TD PORT="#{index}" COLSPAN="{column_size}">Output {index}<BR/>{shape}<BR/>{dtype}<BR/>{layout}<BR/>{memory_config}</TD>
                """
        table += "</TR>"
    table += "</TABLE>>"

    if isinstance(operation, TorchModule):
        color = LEVEL_COLORS[level % len(LEVEL_COLORS)]
        if max_depth is None or level < max_depth - 1:
            with graphviz_graph.subgraph(name=node.name) as cluster_graph:
                cluster_graph.attr(
                    fontcolor="black",
                    bgcolor=color,
                    cluster="true",
                    label=label,
                    rankdir="TB",
                    shape="hexagon",
                )
                cluster_graph.node_attr["style"] = "filled"
                _visualize(
                    cluster_graph,
                    operation.graph,
                    max_depth=max_depth,
                    file_name=None,
                    visualize_node=visualize_node,
                    visualize_edge=visualize_edge,
                    verbose=verbose,
                    level=level + 1,
                )
        else:
            graphviz_graph.node(
                node.name,
                label=table,
                fontcolor="black",
                fillcolor=color,
            )

    else:
        URL = None
        if operation_id is not None:
            URL = f"/operation_buffer_report/{operation_id}"
        graphviz_graph.node(
            node.name,
            label=table,
            fillcolor="#DCDCDC",
            URL=URL,
        )


def visualize(*function_args, file_name=None, visualize_node=visualize_node, **function_kwargs):
    if shutil.which("dot") is None:
        logger.warning("Graphviz is not installed. Skipping visualization.")
        return
    logger.debug(f"Dumping graph of the model to {file_name}")
    return ttnn.torch_tracer.visualize(
        *function_args, file_name=file_name, visualize_node=visualize_node, **function_kwargs
    )


get_graph = ttnn.torch_tracer.get_graph

GRAPH_STACK = None
ENABLE_TRACER = False


def enable_tracing():
    global ENABLE_TRACER
    global GRAPH_STACK
    if ttnn.CONFIG.enable_fast_runtime_mode:
        raise ValueError("Tracing is not supported in fast runtime mode.")
    if ENABLE_TRACER:
        raise ValueError("Tracing is already enabled.")
    ENABLE_TRACER = True
    ttnn.torch_tracer.enable_tracing()
    GRAPH_STACK = ttnn.torch_tracer.GRAPH_STACK


def disable_tracing():
    global ENABLE_TRACER
    global GRAPH_STACK
    ENABLE_TRACER = False
    ttnn.torch_tracer.disable_tracing()
    GRAPH_STACK = ttnn.torch_tracer.GRAPH_STACK


def is_tracing_enabled():
    return ENABLE_TRACER and ttnn.torch_tracer.is_tracing_enabled()


@contextmanager
def trace():
    enable_tracing()
    yield
    disable_tracing()


def get_module_input_nodes(module_operation):
    return [module_input.node for module_input in module_operation.inputs]


def get_module_output_nodes(module_operation):
    return [module_input.node for module_input in module_operation.outputs]


def get_module_input_tensor_names(module_operation):
    module_input_nodes = get_module_input_nodes(module_operation)
    input_tensor_names = []
    for node in module_input_nodes:
        operation = module_operation.graph.nodes[node]["operation"]
        input_tensor_names.append(f"{operation.name}")
    return input_tensor_names


def extract_args_and_kwargs_from_arg_name_value_pairs(arg_name_value_pairs) -> Any:
    function_args = []
    function_kwargs = []
    for arg_name, arg in arg_name_value_pairs:
        if isinstance(arg_name, ttnn.tracer.PositionalArgumentName):
            if not isinstance(arg, ttnn.tracer.InputTensorIndex):
                function_args.append(f"{arg}")
            else:
                function_args.append(f"{arg_name}")
        else:
            if not isinstance(arg, ttnn.tracer.InputTensorIndex):
                function_kwargs.append(f"{arg_name}={arg}")
            else:
                function_kwargs.append(f"{arg_name}={arg_name}")


def node_to_statement(string_io, graph, node, variable, input_variables, prefix):
    def process_arg_name_values(arg_name_values):
        def process_value(value):
            if isinstance(value, ttnn.tracer.InputTensorIndex):
                return input_variables[value.index]
            elif isinstance(value, (tuple, list)):
                joined_elements = ", ".join([process_value(v) for v in value])
                return f"[{joined_elements}]"
            elif isinstance(value, dict):
                joined_elements = ", ".join([f"{key}: {process_value(value)}" for key, value in value.items()])
                return f"{{{joined_elements}}}"
            else:
                return f"{value}"

        return [(name, process_value(value)) for name, value in arg_name_values]

    operation = graph.nodes[node]["operation"]
    shapes = graph.nodes[node]["shapes"]
    dtypes = graph.nodes[node]["dtypes"]
    duration = graph.nodes[node].get("duration", None)

    if isinstance(operation, ttnn.tracer.TorchParameter):
        ttnn_tracer_name = getattr(operation.parameter, "__ttnn_tracer_name__", "")
        ttnn_tracer_name = ttnn_tracer_name.replace(f"{prefix}.", "", 1)
        string_io.write(f"    {variable} = parameters.{ttnn_tracer_name}")
    elif isinstance(operation, ttnn.tracer.TorchTensor):
        string_io.write(
            f"    {variable} = torch.as_tensor({operation.tensor.flatten().tolist()[:8]}, ...).reshape({tuple(operation.tensor.shape)}).to({operation.tensor.dtype})"
        )
    elif isinstance(operation, ttnn.tracer.TorchFunction):
        function_args = []
        function_kwargs = []
        for arg_name, arg in process_arg_name_values(operation.arg_name_value_pairs):
            if isinstance(arg_name, ttnn.tracer.PositionalArgumentName):
                function_args.append(f"{arg}")
            else:
                function_kwargs.append(f"{arg_name}={arg}")

        arguments_string = []
        if function_args:
            arguments_string.append(", ".join(function_args))
        if function_kwargs:
            arguments_string.append(", ".join(function_kwargs))
        arguments_string = ", ".join(arguments_string)

        string_io.write(f"    {variable} = {operation}({arguments_string})")

    elif isinstance(operation, ttnn.tracer.TorchModule):
        module_name = f"{type(operation.module).__name__}"
        ttnn_tracer_name = operation.module.__ttnn_tracer_name__
        ttnn_tracer_name = ttnn_tracer_name.replace(f"{prefix}.", "", 1)

        function_args = []
        function_kwargs = []
        for arg_name, arg in process_arg_name_values(operation.arg_name_value_pairs):
            if isinstance(arg_name, ttnn.tracer.PositionalArgumentName):
                function_args.append(f"{arg}")
            else:
                function_kwargs.append(f"{arg_name}={arg}")

        arguments_string = []
        if function_args:
            arguments_string.append(", ".join(function_args))
        if function_kwargs:
            arguments_string.append(", ".join(function_kwargs))
        arguments_string = ", ".join(arguments_string)

        if ttnn_tracer_name == "":
            string_io.write(f"    {variable} = {module_name}(config, {arguments_string}, parameters=parameters)")
        else:
            string_io.write(
                f"    {variable} = {module_name}(config, {arguments_string}, parameters=parameters.{ttnn_tracer_name})"
            )

    elif isinstance(operation, TTNNOperation):
        function_args = []
        function_kwargs = []
        for arg_name, arg in process_arg_name_values(operation.arg_name_value_pairs):
            if isinstance(arg_name, ttnn.tracer.PositionalArgumentName):
                function_args.append(f"{arg}")
            else:
                function_kwargs.append(f"{arg_name}={arg}")

        arguments_string = []
        if function_args:
            arguments_string.append(", ".join(function_args))
        if function_kwargs:
            arguments_string.append(", ".join(function_kwargs))
        arguments_string = ", ".join(arguments_string)

        string_io.write(f"    {variable} = {operation.pretty_name}({arguments_string})")

    elif isinstance(operation, TTNNTensor):
        string_io.write(f"    {variable} = {operation}")

    else:
        raise ValueError(f"Unknown operation type: {operation}")

    if len(shapes) == 1:
        shapes = shapes[0]
    if len(dtypes) == 1:
        dtypes = dtypes[0]

    string_io.write(f"    # shapes: {shapes}, dtypes: {dtypes}")
    if duration is not None:
        string_io.write(f"; duration: {duration_to_string(duration)}")
    string_io.write("\n")


def module_to_source_code(module_operation, prefix=""):
    graph = module_operation.graph
    string_io = io.StringIO()

    input_tensor_names = get_module_input_tensor_names(module_operation)
    input_tensor_names_as_string = ", ".join(input_tensor_names)

    module_name = f"{type(module_operation.module).__name__}"

    string_io.write(f"def {module_name}(config, {input_tensor_names_as_string}, *, parameters):\n")

    module_input_nodes = get_module_input_nodes(module_operation)
    module_output_nodes = get_module_output_nodes(module_operation)
    node_to_variable = {}
    for module_input, name in zip(module_input_nodes, input_tensor_names):
        node_to_variable[module_input] = name

    index = 0
    for node in nx.topological_sort(graph):
        if node in module_input_nodes:
            continue

        input_nodes = [
            (input_node, edge_data["sink_input_index"]) for input_node, _, edge_data in graph.in_edges(node, data=True)
        ]
        input_nodes = sorted(input_nodes, key=lambda x: x[1])
        input_nodes = [input_node for input_node, _ in input_nodes]
        input_variables = [node_to_variable[input_node] for input_node in input_nodes]

        create_new_variable = True
        if len(input_nodes) == 1 and node not in module_output_nodes:
            input_node = input_nodes[0]
            if len(list(graph.successors(input_node))) == 1:
                create_new_variable = False
                variable = input_variables[0]
                node_to_variable[node] = variable

        if create_new_variable:
            variable = f"variable_{index}"
            index += 1
            node_to_variable[node] = variable

        node_to_statement(string_io, graph, node, variable, input_variables, prefix)

    output_variables = [node_to_variable[output_node] for output_node in module_output_nodes]
    if len(output_variables) == 1:
        string_io.write(f"    return {output_variables[0]}\n")
    else:
        output_variables_as_string = ", ".join(output_variables)
        string_io.write(f"    return {output_variables_as_string}\n")

    return string_io.getvalue()


def codegen_submodules(graph):
    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, ttnn.tracer.TorchModule):
            module = operation.module
            codegen_submodules(operation.graph)
            yield module_to_source_code(operation, module.__ttnn_tracer_name__)


def codegen_top_level_module(graph):
    string_io = io.StringIO()
    string_io.write(f"def top_level_module():\n")

    node_to_variable = {}
    index = 0
    for node in nx.topological_sort(graph):
        input_nodes = [
            (input_node, edge_data["sink_input_index"]) for input_node, _, edge_data in graph.in_edges(node, data=True)
        ]
        input_nodes = sorted(input_nodes, key=lambda x: x[1])
        input_nodes = [input_node for input_node, _ in input_nodes]
        input_variables = [node_to_variable[input_node] for input_node in input_nodes]

        variable = f"variable_{index}"
        index += 1
        node_to_variable[node] = variable
        node_to_statement(string_io, graph, node, variable, input_variables, prefix="")
    return string_io.getvalue()


def codegen(output):
    logger.warning("Codegen is an experimental feature and may not work as expected.")
    graph = get_graph(output)

    output = ""
    for module_code in codegen_submodules(graph):
        output += module_code + "\n\n"
    output += codegen_top_level_module(graph)
    return output
