# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn.operations.golden_common import golden_compute_gradients, golden_prepare_grad_inputs


def _golden_function(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, **_):
    import torch

    # TTNN indices can be uint32, while torch embedding requires signed integer indices.
    output_tensor = torch.nn.functional.embedding(input_tensor.to(torch.int64), weight)
    return output_tensor


ttnn.attach_golden_function(ttnn.embedding, golden_function=_golden_function)


def _golden_function_embedding_bw(input_tensor, weight_tensor, output_gradient_tensor, *_, **__):
    import torch

    # Gradient w.r.t. the embedding weight: scatter the output gradients into the indexed weight rows.
    (weight_tensor,) = golden_prepare_grad_inputs(weight_tensor)
    output = torch.nn.functional.embedding(input_tensor.to(torch.int64), weight_tensor)
    output = output.reshape(output_gradient_tensor.shape)
    return golden_compute_gradients(output, (weight_tensor,), output_gradient_tensor)[0]


ttnn.attach_golden_function(ttnn.embedding_bw, golden_function=_golden_function_embedding_bw)

EmbeddingsType = ttnn._ttnn.operations.embedding.EmbeddingsType

__all__ = []
