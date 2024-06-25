# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import tt_lib as ttl

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.atan2(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()

    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


atan2_bw = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary_backward.atan2_bw)


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.xlogy(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()

    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


xlogy_bw = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary_backward.xlogy_bw)


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.hypot(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()

    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


hypot_bw = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary_backward.hypot_bw)


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.ldexp(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()

    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


ldexp_bw = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary_backward.ldexp_bw)


def _golden_function(grad_tensor, input_tensor, weight_tensor, *args, **kwargs):
    import torch

    batch_size = input_tensor.shape[0]
    no_of_embeddings = input_tensor.shape[1] * input_tensor.shape[2]
    embedding_dim = input_tensor.shape[3]

    input_shape = [batch_size, 1, 1, no_of_embeddings]
    input_index = torch.reshape(torch.arange(0, batch_size * no_of_embeddings), shape=input_shape)
    weights_shape = [batch_size, 1, no_of_embeddings, embedding_dim]
    weights = torch.randn(weights_shape, requires_grad=True)
    grad_shape = [1, 1, batch_size * no_of_embeddings, embedding_dim]
    grad_data = torch.randn(grad_shape, requires_grad=True)

    weights.retain_grad()

    pyt_y = torch.nn.functional.embedding(
        input_index.reshape((batch_size, no_of_embeddings)),
        weights.reshape((batch_size * no_of_embeddings, embedding_dim)),
    ).reshape((1, 1, batch_size * no_of_embeddings, embedding_dim))

    pyt_y.backward(gradient=grad_data)

    golden_output_tensor_a = weights.grad

    return golden_output_tensor_a


embedding_bw = ttnn.register_operation(golden_function=_golden_function)(
    ttnn._ttnn.operations.binary_backward.embedding_bw
)


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.add(input_tensor_a, input_tensor_b, alpha=alpha)
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


addalpha_bw = ttnn.register_operation(golden_function=_golden_function)(
    ttnn._ttnn.operations.binary_backward.addalpha_bw
)


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.add(input_tensor_a, input_tensor_b, alpha=alpha)
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


subalpha_bw = ttnn.register_operation(golden_function=_golden_function)(
    ttnn._ttnn.operations.binary_backward.subalpha_bw
)


def _golden_function(grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch

    pyt_y = torch.add(input_tensor_a, input_tensor_b, alpha=alpha)
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


sub_bw = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.binary_backward.sub_bw)

__all__ = []
