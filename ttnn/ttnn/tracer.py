# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import io
import shutil
import time
from typing import Any

from loguru import logger
import networkx as nx
from pyrsistent import PClass, field

logger.disable("torchtrail")

import torchtrail
from torchtrail.multidigraph import MultiDiGraph, merge_graphs

import ttnn

TracedTorchTensor = torchtrail.tracer.TracedTorchTensor

TorchTensor = torchtrail.tracer.TorchTensor
TorchParameter = torchtrail.tracer.TorchParameter
TorchFunction = torchtrail.tracer.TorchFunction
TorchModule = torchtrail.tracer.TorchModule
TorchModuleInput = torchtrail.tracer.TorchModuleInput

PositionalArgumentName = torchtrail.tracer.PositionalArgumentName
InputTensorIndex = torchtrail.tracer.InputTensorIndex


duration_to_string = torchtrail.tracer.duration_to_string
get_input_tensors = torchtrail.tracer.get_input_tensors
get_arg_name_value_pairs = torchtrail.tracer.get_arg_name_value_pairs


class TTNNTensor(PClass):
    def to_string(self, verbose: bool = False) -> str:
        return "ttnn.Tensor"

    __repr__ = to_string


class TTNNOperation(PClass):
    pretty_name = field(mandatory=True)
    operation = field(mandatory=True)
    arg_name_value_pairs = field(mandatory=True)

    def to_string(self, verbose: bool = False) -> str:
        return self.pretty_name

    __repr__ = to_string


class TracedTTNNTensor(ttnn.Tensor, torchtrail.tracer.TracedTensor):
    def __init__(self, tensor: ttnn.Tensor, *, graph: MultiDiGraph, node: torchtrail.tracer.Node, output_index: int):
        super().__init__(tensor)
        self.graph: MultiDiGraph = graph
        self.node: torchtrail.tracer.Node = node
        self.output_index: int = output_index

    @property
    def name(self) -> str:
        return self.node.name


def create_ttnn_input_tensor(tensor: ttnn.Tensor) -> TracedTTNNTensor:
    unique_id = torchtrail.tracer.get_unique_id()
    node_name = f"ttnn_input_{unique_id}"
    node = torchtrail.tracer.Node(name=node_name, unique_id=unique_id)
    graph = MultiDiGraph().add_node(node, operation=TTNNTensor(), shapes=(tuple(tensor.shape),), dtypes=(tensor.dtype,))
    return TracedTTNNTensor(tensor, graph=graph, node=node, output_index=0)


def preprocess_args_and_kwargs(*function_args, **function_kwargs) -> Any:
    import torch

    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, torchtrail.tracer.TracedTensor):
            return arg
        elif isinstance(arg, ttnn.Tensor):
            return create_ttnn_input_tensor(arg)
        elif isinstance(arg, torch.Tensor):
            return torchtrail.tracer.create_input_tensor(arg)
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
    if isinstance(return_value, torch.Tensor) and not isinstance(return_value, torchtrail.tracer.TracedTorchTensor):
        output_tensors.append(return_value)
    elif isinstance(return_value, ttnn.Tensor) and not isinstance(return_value, TracedTTNNTensor):
        output_tensors.append(create_ttnn_input_tensor(return_value))
    elif isinstance(return_value, torchtrail.tracer.TracedTensor):
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

    if isinstance(return_value, (torch.Tensor, torchtrail.tracer.TracedTorchTensor, ttnn.Tensor, TracedTTNNTensor)):
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
        original_function_args = function_args
        original_function_kwargs = function_kwargs
        function_args, function_kwargs = preprocess_args_and_kwargs(*function_args, **function_kwargs)
        input_tensors = get_input_tensors(function_args) + get_input_tensors(function_kwargs)

        node_name = f"{pretty_operation_name}_{torchtrail.tracer.get_unique_id()}"

        start_time = time.time()
        operation_return_type = operation(*function_args, **function_kwargs)
        end_time = time.time()

        duration = None

        output_tensors = preprocess_return_value(operation_return_type)

        shapes = tuple(tuple(tensor.shape) for tensor in output_tensors)
        dtypes = tuple(tensor.dtype for tensor in output_tensors)

        unique_id = torchtrail.tracer.get_unique_id()
        node_name = f"{pretty_operation_name}_{unique_id}"
        node = torchtrail.tracer.Node(name=node_name, unique_id=unique_id)
        if input_tensors:
            graph = merge_graphs(*((tensor.graph, tensor.node) for tensor in input_tensors))
        else:
            graph = MultiDiGraph()

        try:
            arg_name_value_pairs = get_arg_name_value_pairs(
                operation, function_args=original_function_args, function_kwargs=original_function_kwargs
            )
        except Exception as e:
            arg_name_value_pairs = []

        graph = graph.add_node(
            node,
            operation=TTNNOperation(
                pretty_name=pretty_operation_name, operation=operation, arg_name_value_pairs=arg_name_value_pairs
            ),
            shapes=shapes,
            dtypes=dtypes,
            duration=duration,
        )
        for input_index, tensor in enumerate(input_tensors):
            graph = graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )

        output_tensors = [
            (
                torchtrail.tracer.TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
                if isinstance(tensor, torch.Tensor)
                else TracedTTNNTensor(tensor, graph=graph, node=node, output_index=output_index)
            )
            for output_index, tensor in enumerate(output_tensors)
        ]
        return postprocess_return_value(operation_return_type, output_tensors)

    return call_wrapper


def visualize(*function_args, file_name=None, **function_kwargs):
    if shutil.which("dot") is None:
        logger.warning("Graphviz is not installed. Skipping visualization.")
        return
    logger.info(f"Dumping graph of the model to {file_name}")
    return torchtrail.visualize(*function_args, file_name=file_name, **function_kwargs)


get_graph = torchtrail.get_graph
to_networkx = torchtrail.multidigraph.to_networkx


ENABLE_TRACER = False


@contextmanager
def trace():
    with torchtrail.trace():
        global ENABLE_TRACER

        ENABLE_TRACER = True

        yield

        ENABLE_TRACER = False


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
        torchtrail_name = operation.parameter.torchtrail_name
        torchtrail_name = torchtrail_name.replace(f"{prefix}.", "", 1)
        string_io.write(f"    {variable} = parameters.{torchtrail_name}")
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
        torchtrail_name = operation.module.torchtrail_name
        torchtrail_name = torchtrail_name.replace(f"{prefix}.", "", 1)

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

        if torchtrail_name == "":
            string_io.write(f"    {variable} = {module_name}(config, {arguments_string}, parameters=parameters)")
        else:
            string_io.write(
                f"    {variable} = {module_name}(config, {arguments_string}, parameters=parameters.{torchtrail_name})"
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

    print(string_io.getvalue())


def codegen_submodules(graph):
    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, ttnn.tracer.TorchModule):
            module = operation.module
            codegen_submodules(operation.graph)
            module_to_source_code(operation, module.torchtrail_name)


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
    print(string_io.getvalue())


def codegen(output):
    logger.warning("Codegen is an experimental feature and may not work as expected.")
    graph = get_graph(output)
    codegen_submodules(graph)
    codegen_top_level_module(graph)
