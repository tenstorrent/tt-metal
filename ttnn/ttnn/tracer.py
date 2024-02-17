# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any
from loguru import logger
import shutil

from pyrsistent import PClass, field

logger.disable("torchtrail")

import torchtrail
from torchtrail.multidigraph import MultiDiGraph, merge_graphs, compose_all


import ttnn


class TTNNTensor(PClass):
    def to_string(self, verbose: bool = False) -> str:
        return "ttnn.Tensor"

    __repr__ = to_string


class TTNNOperation(PClass):
    pretty_name = field(mandatory=True)
    operation = field(mandatory=True)

    def to_string(self, verbose: bool = False) -> str:
        return self.pretty_name

    __repr__ = to_string


class TracedTTNNTensor(ttnn.Tensor, torchtrail.tracer.TracedTensor):
    def __init__(self, tensor: ttnn.Tensor, *, graph: MultiDiGraph, node: torchtrail.tracer.Node, output_index: int):
        super().__init__(tensor.value)
        self.graph: MultiDiGraph = graph
        self.node: torchtrail.tracer.Node = node
        self.output_index: int = output_index

    @property
    def name(self) -> str:
        return self.node.name


def create_ttnn_input_tensor(tensor: ttnn.Tensor) -> TracedTTNNTensor:
    node_name = f"ttnn_input_{torchtrail.tracer.get_unique_id()}"
    node = torchtrail.tracer.Node(name=node_name)
    graph = MultiDiGraph().add_node(node, operation=TTNNTensor(), shapes=(tuple(tensor.shape),), dtypes=(tensor.dtype,))
    return TracedTTNNTensor(tensor, graph=graph, node=node, output_index=0)


def preprocess_args_and_kwargs(*args, **kwargs) -> Any:
    import torch

    def preprocess_arg(arg: Any) -> Any:
        if isinstance(arg, torchtrail.tracer.TracedTensor):
            return arg
        elif isinstance(arg, ttnn.Tensor):
            return create_ttnn_input_tensor(arg)
        elif isinstance(arg, torch.Tensor):
            return torchtrail.tracer.create_input_tensor(arg)
        else:
            return arg

    args = [preprocess_arg(arg) for arg in args]
    kwargs = {name: preprocess_arg(arg) for name, arg in kwargs.items()}
    return args, kwargs


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

    def call_wrapper(*args, **kwargs):
        args, kwargs = preprocess_args_and_kwargs(*args, **kwargs)

        input_tensors = [arg for arg in args if isinstance(arg, torchtrail.tracer.TracedTensor)]
        input_tensors += [arg for arg in kwargs.values() if isinstance(arg, torchtrail.tracer.TracedTensor)]

        node_name = f"{pretty_operation_name}_{torchtrail.tracer.get_unique_id()}"

        start_time = time.time()
        operation_return_type = operation(*args, **kwargs)
        end_time = time.time()

        duration = None
        if ttnn.TTNN_ENABLE_LOGGING:
            duration = end_time - start_time

        output_tensors = preprocess_return_value(operation_return_type)

        shapes = tuple(tuple(tensor.shape) for tensor in output_tensors)
        dtypes = tuple(tensor.dtype for tensor in output_tensors)

        node_name = f"{pretty_operation_name}_{torchtrail.tracer.get_unique_id()}"
        node = torchtrail.tracer.Node(name=node_name)
        if input_tensors:
            graph = merge_graphs(*((tensor.graph, tensor.node) for tensor in input_tensors))
        else:
            graph = MultiDiGraph()
        graph = graph.add_node(
            node,
            operation=TTNNOperation(pretty_name=pretty_operation_name, operation=operation),
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
            torchtrail.tracer.TracedTorchTensor(tensor, graph=graph, node=node, output_index=output_index)
            if isinstance(tensor, torch.Tensor)
            else TracedTTNNTensor(tensor, graph=graph, node=node, output_index=output_index)
            for output_index, tensor in enumerate(output_tensors)
        ]
        return postprocess_return_value(operation_return_type, output_tensors)

    return call_wrapper


def visualize(*args, file_name, **kwargs):
    if shutil.which("dot") is None:
        logger.warning("Graphviz is not installed. Skipping visualization.")
        return
    logger.info(f"Dumping graph of the model to {file_name}")
    return torchtrail.visualize(*args, file_name=file_name, **kwargs)


ENABLE_TRACER = False


@contextmanager
def trace():
    with torchtrail.trace():
        global ENABLE_TRACER

        ENABLE_TRACER = True

        yield

        ENABLE_TRACER = False
