# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import inspect
import math
import pathlib
import time
from typing import Any, Callable, Optional, Union, Tuple

import graphviz
import networkx as nx
from loguru import logger
import torch


GRAPH_STACK = None


def enable_tracing():
    global GRAPH_STACK
    if GRAPH_STACK is not None:
        raise RuntimeError("Cannot nest trace calls")

    setattr(torch.nn.Module, "__call__", traced_module_forward)

    for name, op in zip(TORCH_CREATION_OPERATION_NAMES, TORCH_CREATION_OPERATIONS):
        setattr(torch, name, wrap_create_function(op))

    GRAPH_STACK = [nx.MultiDiGraph()]


def disable_tracing():
    global GRAPH_STACK
    # Reset monkey-patched module __call__ and torch creation ops
    setattr(torch.nn.Module, "__call__", TORCH_NN_MODULE_CALL)

    for name, op in zip(TORCH_CREATION_OPERATION_NAMES, TORCH_CREATION_OPERATIONS):
        setattr(torch, name, op)

    GRAPH_STACK = None


def is_tracing_enabled():
    return GRAPH_STACK is not None


TORCH_NN_MODULE_CALL = torch.nn.Module.__call__


# The following functions are overriden to capture input tensors
TORCH_CREATION_OPERATION_NAMES = [
    "as_tensor",
    "from_numpy",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "arange",
    "range",
    "linspace",
    "logspace",
    "eye",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "complex",
    "heaviside",
    "bernoulli",
    "multinomial",
    "normal",
    "poisson",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
]
TORCH_CREATION_OPERATIONS = [getattr(torch, name) for name in TORCH_CREATION_OPERATION_NAMES]


@dataclasses.dataclass
class Node:
    name: str
    unique_id: int

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.unique_id < other.unique_id


@dataclasses.dataclass
class PositionalArgumentName:
    index: int

    def __repr__(self):
        return f"{self.index}"

    def __hash__(self):
        return hash(self.index)


@dataclasses.dataclass
class InputTensorIndex:
    index: int

    def __repr__(self):
        return f"${self.index}"


@dataclasses.dataclass
class TorchTensor:
    tensor: torch.Tensor

    def to_string(self, verbose=False):
        return "torch.Tensor"

    __repr__ = to_string


@dataclasses.dataclass
class TorchParameter:
    parameter: torch.nn.Parameter

    def to_string(self, verbose=False):
        output = "torch.nn.Parameter"
        if hasattr(self.parameter, "__ttnn_tracer_name__"):
            output = f"{output}\n{self.parameter.__ttnn_tracer_name__}"
        return output

    __repr__ = to_string


@dataclasses.dataclass
class TorchFunction:
    function: Any
    arg_name_value_pairs: list

    def to_string(self, verbose=False):
        if "__module__" in dir(self.function):
            output = f"{self.function.__module__}.{self.function.__name__}"
        elif "__objclass__" in dir(self.function):
            output = f"{self.function.__objclass__.__module__}.{ self.function.__objclass__.__name__}.{self.function.__name__}"
        elif "__class__" in dir(self.function):
            output = f"{self.function.__class__}.{self.function.__name__}"
        else:
            raise RuntimeError(f"Unknown function type: {type(self.function)}")
        output = output.replace("torch.nn.modules", "torch.nn")
        output = output.replace("torch._tensor", "torch.Tensor")
        output = output.replace("torch._C.TensorBase", "torch.Tensor")
        output = output.replace("torch._C._nn", "torch.nn.functional")
        output = output.replace("torch._C", "torch")

        if not verbose:
            return output

        output += "("
        current_length = len(output)
        if current_length > 50:
            output += "\n"
            current_length = 0
        for index, (name, value) in enumerate(self.arg_name_value_pairs):
            if isinstance(value, torch.Tensor):
                value = f"torch.Tensor(...)"
            elif isinstance(name, PositionalArgumentName):
                value_as_str = f"{value}"
            else:
                value_as_str = f"{name}={value}"
            current_length += len(value_as_str)
            output += value_as_str
            if index != len(self.arg_name_value_pairs) - 1:
                if current_length < 50:
                    output += ", "
                else:
                    output += ",\n"
                    current_length = 0

        output += ")"
        return output

    __repr__ = to_string


@dataclasses.dataclass
class TorchModule:
    module: torch.nn.Module
    graph: nx.MultiDiGraph
    inputs: list[TracedTensor]
    outputs: list[TracedTensor]
    arg_name_value_pairs: list

    def to_string(self, verbose=False):
        output = f"{type(self.module).__module__}.{type(self.module).__name__}"

        if verbose:
            output += "("
            current_length = len(output)
            if current_length > 50:
                output += "\n"
                current_length = 0
            for index, (name, value) in enumerate(self.arg_name_value_pairs):
                if isinstance(name, PositionalArgumentName):
                    value_as_str = f"{value}"
                else:
                    value_as_str = f"{name}={value}"

                current_length += len(value_as_str)
                output += value_as_str
                if index != len(self.arg_name_value_pairs) - 1:
                    if current_length < 50:
                        output += ", "
                    else:
                        output += ",\n"
                        current_length = 0

            output += ")"

        if self.module.__ttnn_tracer_name__ != "":
            output += f"\n{self.module.__ttnn_tracer_name__}"
        return output

    __repr__ = to_string


@dataclasses.dataclass
class TorchModuleInput:
    name: str

    def to_string(self, verbose=False):
        return f"{self.name}"

    __repr__ = to_string


class ModuleIOTensor:
    def __init__(self, graph: nx.MultiDiGraph, node: Node, output_index: int):
        self.graph: nx.MultiDiGraph = graph
        self.node: Node = node
        self.output_index: int = output_index

    @classmethod
    def from_traced_tensor(cls, traced_tensor: TracedTorchTensor):
        return cls(traced_tensor.graph, traced_tensor.node, traced_tensor.output_index)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> str:
        return self.graph.nodes[self.node]["shapes"][self.output_index]


UNIQUE_ID = 0


def get_unique_id():
    global UNIQUE_ID
    output = UNIQUE_ID
    UNIQUE_ID += 1
    return output


def create_input_tensor(
    tensor: torch.Tensor,
    function: Optional[Callable[..., Any]] = None,
    arg_name_value_pairs=None,
    duration=None,
) -> TracedTorchTensor:
    if isinstance(tensor, torch.nn.Parameter):
        unique_id = get_unique_id()
        node_name = f"torch_parameter_{unique_id}"
        node = Node(name=node_name, unique_id=unique_id)
        graph = GRAPH_STACK[-1]
        graph.add_node(
            node,
            operation=TorchParameter(parameter=tensor),
            shapes=(tuple(tensor.shape),),
            dtypes=(tensor.dtype,),
            duration=duration,
        )
        return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)
    elif isinstance(tensor, torch.Tensor):
        if function is None:
            operation = TorchTensor(tensor=tensor)
        else:
            arg_name_value_pairs = arg_name_value_pairs if arg_name_value_pairs is not None else {}
            operation = TorchFunction(function=function, arg_name_value_pairs=arg_name_value_pairs)

        unique_id = get_unique_id()
        node_name = f"torch_input_{unique_id}"
        node = Node(name=node_name, unique_id=unique_id)

        graph = GRAPH_STACK[-1]
        graph.add_node(
            node,
            operation=operation,
            shapes=(tuple(tensor.shape),),
            dtypes=(tensor.dtype,),
            duration=duration,
        )
        return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)
    else:
        raise RuntimeError(f"Unknown input type: {type(tensor)}")


def preprocess_args_and_kwargs(*function_args, **function_kwargs) -> Any:
    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, TracedTensor):
            return arg
        elif isinstance(arg, torch.Tensor):
            return create_input_tensor(arg)
        elif isinstance(arg, (tuple, list)):
            return type(arg)([preprocess_arg(element) for element in arg])
        elif isinstance(arg, dict):
            return {key: preprocess_arg(value) for key, value in arg.items()}
        else:
            return arg

    function_args = [preprocess_arg(arg) for arg in function_args]
    function_kwargs = {name: preprocess_arg(arg) for name, arg in function_kwargs.items()}
    return function_args, function_kwargs


def get_input_tensors(object):
    input_tensors = []
    if isinstance(object, TracedTensor):
        input_tensors.append(object)
    elif isinstance(object, (list, tuple)):
        for element in object:
            input_tensors += get_input_tensors(element)
    elif isinstance(object, dict):
        for value in object.values():
            input_tensors += get_input_tensors(value)
    return input_tensors


def get_arg_names(function, *, function_args, function_kwargs):
    if inspect.isbuiltin(function):
        arg_names = [PositionalArgumentName(index=index) for index in range(len(function_args))] + list(
            function_kwargs.keys()
        )
    elif inspect.ismethoddescriptor(function):
        arg_names = [PositionalArgumentName(index=index) for index in range(len(function_args))] + list(
            function_kwargs.keys()
        )
    else:
        signature = inspect.signature(function)
        arg_names = [parameter.name for parameter in signature.parameters.values()]
        if inspect.ismethod(function):
            arg_names = ["self"] + arg_names
        if len(arg_names) < len(function_args) + len(function_kwargs):
            arg_names = [PositionalArgumentName(index=index) for index in range(len(function_args))] + list(
                function_kwargs.keys()
            )

    if len(arg_names) < len(function_args) + len(function_kwargs):
        raise RuntimeError(f"Number of argument names must be at least as large as the number of arguments")
    return arg_names


def get_arg_name_value_pairs(function, *, function_args, function_kwargs):
    arg_names = get_arg_names(function, function_args=function_args, function_kwargs=function_kwargs)

    try:
        signature = inspect.signature(function)
    except:
        signature = None

    input_tensor_index = 0

    def process_arg_value(arg_value):
        nonlocal input_tensor_index
        if isinstance(arg_value, TracedTensor):
            output = InputTensorIndex(index=input_tensor_index)
            input_tensor_index += 1
            return output
        elif isinstance(arg_value, (tuple, list)):
            return type(arg_value)([process_arg_value(element) for element in arg_value])
        elif isinstance(arg_value, dict):
            return {key: process_arg_value(value) for key, value in arg_value.items()}
        else:
            return arg_value

    arg_name_value_pairs = []
    for arg_name, arg_value in zip(arg_names, function_args):
        if signature is not None:
            if (
                arg_name in signature.parameters
                and signature.parameters[arg_name].default != inspect.Parameter.empty
                and arg_value == signature.parameters[arg_name].default
            ):
                continue
        arg_name_value_pairs.append((arg_name, process_arg_value(arg_value)))

    for arg_name in arg_names[len(function_args) :]:
        if arg_name not in function_kwargs:
            continue
        arg_value = function_kwargs[arg_name]

        if signature is not None:
            if (
                arg_name in signature.parameters
                and signature.parameters[arg_name].default != inspect.Parameter.empty
                and arg_value == signature.parameters[arg_name].default
            ):
                continue
        arg_name_value_pairs.append((arg_name, process_arg_value(arg_value)))

    return arg_name_value_pairs


def preprocess_return_value(return_value):
    output_tensors = []
    if isinstance(
        return_value,
        (
            int,
            torch.Size,
            torch.device,
            torch.dtype,
            str,
        ),
    ):
        pass
    elif isinstance(return_value, TracedTensor):
        output_tensors.append(return_value)
    elif isinstance(return_value, torch.Tensor):
        output_tensors.append(create_input_tensor(return_value))
    elif isinstance(return_value, (tuple, list)):
        for value in return_value:
            output_tensors += preprocess_return_value(value)
    elif dataclasses.is_dataclass(return_value):
        for class_field in dataclasses.fields(return_value):
            value = getattr(return_value, class_field.name)
            output_tensors += preprocess_return_value(value)
    elif isinstance(return_value, dict):
        for value in return_value.values():
            output_tensors += preprocess_return_value(value)
    elif return_value is None:
        pass
    else:
        logger.warning(f"preprocess_return_value: unsupported type {type(return_value)}")
    return output_tensors


def postprocess_return_value(return_value, output_tensors):
    if isinstance(return_value, TracedTensor):
        output_tensor, *_ = output_tensors
        output_tensors.pop(0)
        return output_tensor
    elif isinstance(return_value, torch.Tensor):
        output_tensor, *_ = output_tensors
        output_tensors.pop(0)
        return output_tensor
    elif isinstance(return_value, (tuple, list)):
        return type(return_value)(postprocess_return_value(value, output_tensors) for value in return_value)
    elif dataclasses.is_dataclass(return_value):
        updated_fields = {}
        for class_field in dataclasses.fields(return_value):
            value = getattr(return_value, class_field.name)
            updated_fields[class_field.name] = postprocess_return_value(value, output_tensors)
        return type(return_value)(**updated_fields)
    elif isinstance(return_value, dict):
        return {name: postprocess_return_value(value, output_tensors) for name, value in return_value.items()}
    else:
        return return_value


class TracedTensor:
    ...


class TracedTorchTensor(torch.Tensor, TracedTensor):
    @staticmethod
    def __new__(
        cls: Any,
        tensor: Any,
        graph: nx.MultiDiGraph,
        node: Node,
        output_index: int,
        *function_args: Any,
        **function_kwargs: Any,
    ) -> Any:
        return super().__new__(cls, tensor, *function_args, **function_kwargs)  # type: ignore[call-arg]

    def __init__(
        self,
        tensor: Any,
        *,
        graph: nx.MultiDiGraph,
        node: Node,
        output_index: int,
    ):
        self.graph: nx.MultiDiGraph = graph
        self.node: Node = node
        self.output_index: int = output_index

    @property
    def name(self) -> str:
        return self.node.name

    @classmethod
    def __torch_function__(
        cls: Any,
        function,
        types: Any,
        function_args: Any = (),
        function_kwargs: Any = None,
    ) -> Any:
        types = tuple(torch.Tensor if t == TorchTensor else t for t in types)

        if not is_tracing_enabled():
            return super().__torch_function__(function, types, function_args, function_kwargs)

        if function_kwargs is None:
            function_kwargs = {}

        function_args, function_kwargs = preprocess_args_and_kwargs(*function_args, **function_kwargs)
        input_tensors = get_input_tensors(function_args) + get_input_tensors(function_kwargs)

        start_time = time.time()
        function_return_value = super().__torch_function__(function, types, function_args, function_kwargs)
        end_time = time.time()
        duration = end_time - start_time

        output_tensors = preprocess_return_value(function_return_value)
        if not output_tensors:
            return function_return_value

        shapes = tuple(tuple(tensor.shape) for tensor in output_tensors)
        dtypes = tuple(tensor.dtype for tensor in output_tensors)

        arg_name_value_pairs = get_arg_name_value_pairs(
            function, function_args=function_args, function_kwargs=function_kwargs
        )

        unique_id = get_unique_id()
        node_name = f"{function.__name__}_{unique_id}"
        node = Node(name=node_name, unique_id=unique_id)

        graph = GRAPH_STACK[-1]
        for tensor in input_tensors:
            if tensor.graph is not graph:
                graph = nx.compose(graph, tensor.graph)
                GRAPH_STACK[-1] = graph

        graph.add_node(
            node,
            operation=TorchFunction(function=function, arg_name_value_pairs=arg_name_value_pairs),
            shapes=shapes,
            dtypes=dtypes,
            duration=duration,
        )
        for input_index, tensor in enumerate(input_tensors):
            graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )

        output_tensors = [
            TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
            for output_index, tensor in enumerate(output_tensors)
        ]
        return postprocess_return_value(function_return_value, output_tensors)


def wrap_create_function(function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*function_args: Any, **function_kwargs: Any) -> TracedTensor:
        arg_name_value_pairs = get_arg_name_value_pairs(
            function, function_args=function_args, function_kwargs=function_kwargs
        )
        start_time = time.time()
        input_tensor = function(*function_args, **function_kwargs)
        end_time = time.time()
        duration = end_time - start_time
        return create_input_tensor(
            input_tensor,
            function,
            arg_name_value_pairs=arg_name_value_pairs,
            duration=duration,
        )

    return wrapper


def create_module_input(name, tensor: torch.Tensor) -> TracedTensor:
    unique_id = get_unique_id()
    node_name = f"module_input_{unique_id}"
    node = Node(name=node_name, unique_id=unique_id)
    graph = GRAPH_STACK[-1]
    graph.add_node(
        node,
        operation=TorchModuleInput(name=name),
        shapes=(tuple(tensor.shape),),
        dtypes=(tensor.dtype,),
    )
    return TracedTorchTensor(tensor, graph=graph, node=node, output_index=0)


def convert_to_module_args_and_kwargs(module, *function_args, **function_kwargs) -> Any:
    def preprocess_arg(name: str, arg: Any) -> Any:
        if isinstance(arg, TracedTensor):
            output = create_module_input(name, arg)
            return output
        elif isinstance(arg, torch.nn.Parameter):
            raise RuntimeError("Module parameters are not supported")
        elif isinstance(arg, (tuple, list)):
            return type(arg)([preprocess_arg(name, element) for element in arg])
        elif isinstance(arg, dict):
            return {key: preprocess_arg(name, element) for key, element in arg.items()}
        else:
            return arg

    arg_names = get_arg_names(module.forward, function_args=function_args, function_kwargs=function_kwargs)
    function_args = [preprocess_arg(name, arg) for name, arg in zip(arg_names, function_args)]
    function_kwargs = {name: preprocess_arg(name, arg) for name, arg in function_kwargs.items()}
    return function_args, function_kwargs


def create_module(module, module_input_tensors, module_output_tensors, arg_name_value_pairs):
    module_inputs = [ModuleIOTensor.from_traced_tensor(tensor) for tensor in module_input_tensors]
    module_outputs = [ModuleIOTensor.from_traced_tensor(tensor) for tensor in module_output_tensors]
    module_graph = GRAPH_STACK[-1]
    operation = TorchModule(
        module=module,
        graph=module_graph,
        inputs=module_inputs,
        outputs=module_outputs,
        arg_name_value_pairs=arg_name_value_pairs,
    )
    return operation


def set___ttnn_tracer_name__(module, name):
    if hasattr(module, "__ttnn_tracer_name__"):
        return
    module.__ttnn_tracer_name__ = name

    if isinstance(module, torch.nn.ModuleList):
        for index, child in enumerate(module.children()):
            set___ttnn_tracer_name__(child, f"{name}.{index}" if name else f"{index}")

    for child_name, child in module.named_children():
        set___ttnn_tracer_name__(child, f"{name}.{child_name}" if name else child_name)

    for parameter_name, parameter in module.named_parameters():
        parameter.__ttnn_tracer_name__ = parameter_name


def traced_module_forward(*function_args: Any, **function_kwargs: Any) -> Any:
    if not is_tracing_enabled():
        return TORCH_NN_MODULE_CALL(*function_args, **function_kwargs)

    module = function_args[0]

    set___ttnn_tracer_name__(module, "")

    function_args, function_kwargs = preprocess_args_and_kwargs(*function_args, **function_kwargs)

    GRAPH_STACK.append(nx.MultiDiGraph())
    module_args, module_kwargs = convert_to_module_args_and_kwargs(module, *function_args, **function_kwargs)

    start_time = time.time()
    module_return_value = TORCH_NN_MODULE_CALL(*module_args, **module_kwargs)
    end_time = time.time()
    duration = end_time - start_time

    module_input_tensors = get_input_tensors(module_args) + get_input_tensors(module_kwargs)
    module_output_tensors = preprocess_return_value(module_return_value)

    shapes = tuple(tuple(tensor.shape) for tensor in module_output_tensors)
    dtypes = tuple(tensor.dtype for tensor in module_output_tensors)

    arg_name_value_pairs = get_arg_name_value_pairs(
        module.forward, function_args=function_args, function_kwargs=function_kwargs
    )[1:]
    operation = create_module(module, module_input_tensors, module_output_tensors, arg_name_value_pairs)
    GRAPH_STACK.pop()

    unique_id = get_unique_id()
    node_name = f"{module.__ttnn_tracer_name__}_{unique_id}"
    node = Node(name=node_name, unique_id=unique_id)

    input_tensors = get_input_tensors(function_args) + get_input_tensors(function_kwargs)

    graph = GRAPH_STACK[-1]
    for tensor in input_tensors:
        if tensor.graph is not graph:
            graph = nx.compose(graph, tensor.graph)
            GRAPH_STACK[-1] = graph

    graph.add_node(
        node,
        operation=operation,
        shapes=shapes,
        dtypes=dtypes,
        duration=duration,
    )

    for input_index, tensor in enumerate(input_tensors):
        graph.add_edge(
            tensor.node,
            node,
            source_output_index=tensor.output_index,
            sink_input_index=input_index,
        )

    output_tensors = [
        TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
        for output_index, tensor in enumerate(module_output_tensors)
    ]
    return postprocess_return_value(module_return_value, output_tensors)


@contextmanager
def trace():
    try:
        enable_tracing()
        yield
    finally:
        disable_tracing()


LEVEL_COLORS = [
    "#0000ff80",
    "#ee00ee80   ",
    "#ff000080",
    "#eeee0080",
    "#00ff0080",
    "#00eeee80",
]


def get_source(graph, node, source_output_index, level, max_depth=None):
    operation = graph.nodes[node]["operation"]

    if max_depth is not None and level + 1 == max_depth:
        return node, source_output_index, graph.nodes[node]

    if not isinstance(operation, TorchModule):
        return node, source_output_index, graph.nodes[node]

    module = operation
    module_graph = module.graph
    module_node = module.outputs[source_output_index].node
    module_node_output_index = module.outputs[source_output_index].output_index
    return get_source(
        module_graph,
        module_node,
        module_node_output_index,
        level=level + 1,
        max_depth=max_depth,
    )


def get_sink(graph, node, sink_input_index, level, max_depth=None):
    operation = graph.nodes[node]["operation"]

    if not isinstance(operation, TorchModule):
        return node, sink_input_index, graph.nodes[node]

    if max_depth is not None and level + 1 == max_depth:
        return node, sink_input_index, graph.nodes[node]

    module = operation
    module_graph = module.graph
    module_node = module.inputs[sink_input_index].node
    return get_sink(
        module_graph,
        module_node,
        0,  # module_node is always a torchtrail.tracer.TorchModuleInput
        level=level + 1,
        max_depth=max_depth,
    )


def duration_to_string(duration):
    if duration < 1e-6:
        return f"{duration * 1e9:.1f} ns"
    elif duration < 1e-3:
        return f"{duration * 1e6:.1f} µs"
    elif duration < 1:
        return f"{duration * 1e3:.1f} ms"
    else:
        return f"{duration:.1f} s"


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

    input_tensors = []
    for source_node, _, edge_data in graph.in_edges(node, data=True):
        input_shape = graph.nodes[source_node]["shapes"][edge_data["source_output_index"]]
        input_dtype = graph.nodes[source_node]["dtypes"][edge_data["source_output_index"]]
        input_tensors.append((input_shape, input_dtype))

    output_shapes = attributes["shapes"]
    output_dtypes = attributes["dtypes"]
    output_tensors = tuple((shape, dtype) for shape, dtype in zip(output_shapes, output_dtypes))

    label = operation.to_string(verbose=verbose)

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
            column_size = input_column_sizes[index]
            table = (
                table
                + f"""
                    <TD PORT="${index}" COLSPAN="{column_size}">Input {index}<BR/>{shape}<BR/>{dtype}</TD>
                """
            )
    table += "</TR>"

    if output_tensors:
        table += "<TR>"
        output_column_sizes = compute_even_column_sizes(num_columns, len(output_tensors))
        for index, (shape, dtype) in enumerate(output_tensors):
            column_size = output_column_sizes[index]
            table += f"""
                    <TD PORT="#{index}" COLSPAN="{column_size}">Output {index}<BR/>{shape}<BR/>{dtype}</TD>
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
        graphviz_graph.node(
            node.name,
            label=table,
            fillcolor="#DCDCDC",
        )


def visualize_edge(graphviz_graph, graph, edge, max_depth, level):
    source_node, sink_node, _, edge_data = edge

    source_output_index = edge_data["source_output_index"]
    sink_input_index = edge_data["sink_input_index"]

    source_node, source_output_index, *_ = get_source(
        graph,
        source_node,
        source_output_index,
        level,
        max_depth=max_depth,
    )

    sink_node, sink_input_index, sink_attributes = get_sink(
        graph,
        sink_node,
        sink_input_index,
        level,
        max_depth=max_depth,
    )

    source_name = f"{source_node.name}:#{source_output_index}"
    sink_name = f"{sink_node.name}:${sink_input_index}"
    if isinstance(sink_attributes["operation"], TorchModuleInput):
        sink_name = sink_node.name

    graphviz_graph.edge(
        source_name,
        sink_name,
        label=f"{source_output_index} -> {sink_input_index}",
        fontcolor="black" if level == 0 else "white",
    )


def _visualize(
    graphviz_graph,
    graph,
    *,
    max_depth,
    file_name,
    visualize_node,
    visualize_edge,
    verbose,
    level=0,
) -> graphviz.Digraph:
    if max_depth is not None:
        if max_depth < 1:
            raise RuntimeError("max_depth must be greater than 0")
        if level == max_depth:
            return graphviz_graph

    for node in sorted(graph):
        visualize_node(
            graphviz_graph,
            graph,
            node,
            max_depth,
            visualize_node,
            visualize_edge,
            level,
            verbose=verbose,
        )

    for node in sorted(graph):
        for edge in graph.in_edges(node, data=True, keys=True):
            visualize_edge(graphviz_graph, graph, edge, max_depth, level)

    if file_name is not None:
        file_name = pathlib.Path(file_name)
        if file_name.suffix not in {".svg", ".png", ".pdf"}:
            raise ValueError(f"file_name must have a .svg, .png or .pdf suffix, not {file_name.suffix}")
        format = file_name.suffix[1:]
        graphviz_graph.render(file_name.with_suffix(""), format=format)
        logger.info(f'Graph visualization saved to "{file_name}"')

    return graphviz_graph


def _flatten_graph(
    graph,
    *,
    new_graph=None,
    level=0,
) -> nx.MultiDiGraph:
    if new_graph is None:
        new_graph = nx.MultiDiGraph()

    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, TorchModule):
            module_graph = _flatten_graph(operation.graph, new_graph=new_graph, level=level + 1)
            new_graph = nx.compose_all([new_graph, module_graph])
        else:
            new_graph.add_node(
                node,
                **graph.nodes[node],
            )

    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        for source_node, sink_node, edge_data in graph.in_edges(node, data=True):
            source_node, source_output_index, _ = get_source(
                graph, source_node, edge_data["source_output_index"], level
            )
            sink_node, sink_input_index, _ = get_sink(graph, sink_node, edge_data["sink_input_index"], level)
            new_graph.add_edge(
                source_node,
                sink_node,
                source_output_index=source_output_index,
                sink_input_index=sink_input_index,
            )

    return new_graph


def _remove_module_forward_args(graph):
    new_graph = nx.MultiDiGraph()

    for node in graph:
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, TorchModuleInput):
            continue
        new_graph.add_node(
            node,
            **graph.nodes[node],
        )

    for node in nx.topological_sort(graph):
        operation = graph.nodes[node]["operation"]
        if isinstance(operation, TorchModuleInput):
            continue
        for source_node, _, edge_data in graph.in_edges(node, data=True):
            while isinstance(graph.nodes[source_node]["operation"], TorchModuleInput):
                ((source_node, _, source_node_edge_data),) = graph.in_edges(source_node, data=True)
                edge_data["source_output_index"] = source_node_edge_data["source_output_index"]
            new_graph.add_edge(
                source_node,
                node,
                **edge_data,
            )

    return new_graph


def flatten_graph(graph) -> nx.MultiDiGraph:
    graph = _flatten_graph(graph)
    graph = _remove_module_forward_args(graph)
    return graph


def process_output(output):
    output_tensors = []
    if isinstance(output, TracedTensor):
        output_tensors.append(output)
    elif isinstance(output, (tuple, list)):
        for value in output:
            output_tensors += process_output(value)
    elif dataclasses.is_dataclass(output):
        for class_field in dataclasses.fields(output):
            value = getattr(output, class_field.name)
            output_tensors += process_output(value)
    elif isinstance(output, dict):
        for value in output.values():
            output_tensors += process_output(value)
    elif output is None:
        pass
    else:
        raise RuntimeError(f"Unexpected type {type(output)}")
    return output_tensors


def get_graph(
    value: Union[nx.MultiDiGraph, TracedTensor, Tuple[TracedTensor, ...]],
    flatten: bool = False,
) -> nx.MultiDiGraph:
    if isinstance(value, nx.MultiDiGraph):
        return value
    output_tensors = process_output(value)
    graph = nx.compose_all([tensor.graph for tensor in output_tensors])

    if flatten:
        graph = flatten_graph(graph)
    return graph


def visualize(
    output: Union[TracedTensor, Tuple[TracedTensor, ...]],
    *,
    show_modules: bool = True,
    max_depth: Optional[int] = None,
    file_name: Optional[Union[str, pathlib.Path]] = None,
    graph_attr: Optional[dict] = None,
    node_attr: Optional[dict] = None,
    edge_attr: Optional[dict] = None,
    visualize_node: Callable = visualize_node,
    visualize_edge: Callable = visualize_edge,
    verbose: bool = False,
) -> graphviz.Digraph:
    if not show_modules and max_depth is not None:
        raise RuntimeError("max_depth is not supported with show_modules=True")

    graph = get_graph(output)
    if not show_modules:
        graph = flatten_graph(graph)

    if graph_attr is None:
        graph_attr = {"ordering": "in", "rankdir": "TB"}

    if node_attr is None:
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
    if edge_attr is None:
        edge_attr = {
            "fontsize": "10",
        }

    graphviz_graph = graphviz.Digraph(
        engine="dot",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )

    return _visualize(
        graphviz_graph,
        graph,
        max_depth=max_depth,
        file_name=file_name,
        visualize_node=visualize_node,
        visualize_edge=visualize_edge,
        verbose=verbose,
    )
