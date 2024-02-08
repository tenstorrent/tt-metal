# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Union, Tuple, Optional
from contextlib import contextmanager

import graphviz
from pyrsistent import PClass, field

import torchtrail
from torchtrail.tracer import (
    Node,
    TorchTensor,
    TracedTorchTensor,
    get_unique_id,
    create_input_tensor,
    _visualize,
    flatten_graph,
)
from torchtrail.multidigraph import MultiDiGraph, merge_graphs, compose_all


import ttnn


class TTNNTensor(PClass):
    tensor = field(mandatory=True)

    def __repr__(self):
        return "ttnn.Tensor"


class TTNNOperation(PClass):
    pretty_name = field(mandatory=True)
    operation = field(mandatory=True)

    def __repr__(self):
        return self.pretty_name


class TracedTTNNTensor(ttnn.Tensor):
    def __init__(self, tensor: ttnn.Tensor, *, graph: MultiDiGraph, node: Node, output_index: int):
        super().__init__(tensor.value)
        self.graph: MultiDiGraph = graph
        self.node: Node = node
        self.output_index: int = output_index

    @property
    def name(self) -> str:
        return self.node.name


def create_ttnn_input_tensor(tensor: ttnn.Tensor) -> TracedTTNNTensor:
    node_name = f"ttnn_input_{get_unique_id()}"
    node = Node(name=node_name)
    graph = MultiDiGraph().add_node(
        node, operation=TTNNTensor(tensor=tensor), shapes=(tuple(tensor.shape),), dtypes=(tensor.dtype,)
    )
    return TracedTTNNTensor(tensor, graph=graph, node=node, output_index=0)


def preprocess_args_and_kwargs(*args, **kwargs) -> Any:
    import torch

    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, (TracedTTNNTensor, TracedTorchTensor)):
            return arg
        elif isinstance(arg, ttnn.Tensor):
            return create_ttnn_input_tensor(arg)
        elif isinstance(arg, torch.Tensor):
            return create_input_tensor(arg)
        else:
            return arg

    args = [preprocess_arg(arg) for arg in args]
    kwargs = {name: preprocess_arg(arg) for name, arg in kwargs.items()}
    return args, kwargs


def preprocess_return_value(return_value):
    import torch

    output_tensors = []
    if isinstance(return_value, torch.Tensor) and not isinstance(return_value, TracedTorchTensor):
        output_tensors.append(return_value)
    elif isinstance(return_value, ttnn.Tensor) and not isinstance(return_value, TracedTTNNTensor):
        output_tensors.append(create_ttnn_input_tensor(return_value))
    elif isinstance(return_value, (TracedTorchTensor, TracedTTNNTensor)):
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

    if isinstance(return_value, (torch.Tensor, TracedTorchTensor, ttnn.Tensor, TracedTTNNTensor)):
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

    def call_wrapper(*args, **kwargs):
        args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)

        input_tensors = [arg for arg in args if isinstance(arg, (TracedTTNNTensor, TracedTorchTensor))]
        input_tensors += [arg for arg in kwargs.values() if isinstance(arg, (TracedTTNNTensor, TracedTorchTensor))]

        node_name = f"{pretty_operation_name}_{get_unique_id()}"

        operation_return_type = operation(*args, **kwargs)

        output_tensors = preprocess_return_value(operation_return_type)

        shapes = tuple(tuple(tensor.shape) for tensor in output_tensors)
        dtypes = tuple(tensor.dtype for tensor in output_tensors)

        node_name = f"{pretty_operation_name}_{get_unique_id()}"
        node = Node(name=node_name)
        if input_tensors:
            graph = merge_graphs(*((tensor.graph, tensor.node) for tensor in input_tensors))
        else:
            graph = MultiDiGraph()
        graph = graph.add_node(
            node,
            operation=TTNNOperation(pretty_name=pretty_operation_name, operation=operation),
            shapes=shapes,
            dtypes=dtypes,
        )
        for input_index, tensor in enumerate(input_tensors):
            graph = graph.add_edge(
                tensor.node,
                node,
                source_output_index=tensor.output_index,
                sink_input_index=input_index,
            )

        output_tensors = [
            TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
            if isinstance(tensor, torch.Tensor)
            else TracedTTNNTensor(tensor, graph=graph, node=node, output_index=output_index)
            for output_index, tensor in enumerate(output_tensors)
        ]
        return postprocess_return_value(operation_return_type, output_tensors)

    return call_wrapper


def visualize(
    value: Union[TracedTorchTensor, Tuple[TracedTorchTensor, ...]],
    *,
    show_modules: bool = True,
    max_depth: Optional[int] = None,
    file_name: Optional[str] = None,
) -> graphviz.Digraph:
    if not show_modules and max_depth is not None:
        raise ValueError("max_depth is not supported with show_modules=True")

    if isinstance(value, (TracedTorchTensor, TracedTTNNTensor)):
        graph = value.graph
    elif isinstance(value, tuple):
        graph = compose_all(*[tensor.graph for tensor in value])
    else:
        raise ValueError(f"Unexpected input type: {type(value)}")

    if not show_modules:
        graph = flatten_graph(graph)
    return _visualize(graph, max_depth=max_depth, file_name=file_name)


ENABLE_TRACER = False


@contextmanager
def trace():
    import torch

    with torchtrail.trace(), torch.no_grad():
        global ENABLE_TRACER

        ENABLE_TRACER = True

        yield

        ENABLE_TRACER = False
