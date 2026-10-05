# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import ttnn
from ttnn.decorators import get_golden_function
from ttnn.operations.activations import get_golden_function_for_activation

MatmulProgramConfig = ttnn._ttnn.operations.matmul.MatmulProgramConfig
MatmulMultiCoreReuseProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseProgramConfig
MatmulMultiCoreReuseMultiCastProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastProgramConfig
MatmulMultiCoreReuseMultiCast1DProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCast1DProgramConfig
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig = (
    ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
)
MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig = (
    ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig
)
MatmulParams = ttnn._ttnn.operations.matmul.MatmulParams
MatmulInputs = ttnn._ttnn.operations.matmul.MatmulInputs
MatmulDeviceOperation = ttnn._ttnn.operations.matmul.MatmulDeviceOperation
create_matmul_attributes = ttnn._ttnn.operations.matmul.create_matmul_attributes
matmul_select_program_factory = ttnn._ttnn.operations.matmul.matmul_select_program_factory


def _golden_function(
    input_tensor_a,
    input_tensor_b,
    transpose_a=False,
    transpose_b=False,
    *,
    bias=None,
    activation=None,
    program_config=None,
    **kwargs,
):
    import torch

    if transpose_a:
        input_tensor_a = input_tensor_a.transpose(-1, -2)
    if transpose_b:
        input_tensor_b = input_tensor_b.transpose(-1, -2)
    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    # First check if there is a fused activation in the program config
    if program_config is not None and hasattr(program_config, "fused_activation") and program_config.fused_activation:
        program_config_activation = program_config.fused_activation.op_type
        output_tensor = get_golden_function_for_activation(program_config_activation)(output_tensor)

    # Do the composite op activation function if it is requested
    elif activation is not None:
        output_tensor = get_golden_function_for_activation(activation)(output_tensor)

    while len(output_tensor.shape) > len(input_tensor_a.shape):
        output_tensor = output_tensor.squeeze(0)
    return output_tensor


ttnn.attach_golden_function(
    ttnn.matmul,
    golden_function=_golden_function,
)


def _golden_function(
    input_tensor_a,
    input_tensor_b,
    transpose_a=False,
    transpose_b=False,
    *,
    bias=None,
    program_config=None,
    activation=None,
    **kwargs,
):
    import torch

    if transpose_a:
        input_tensor_a = input_tensor_a.transpose(-1, -2)
    if transpose_b:
        input_tensor_b = input_tensor_b.transpose(-1, -2)
    output_tensor = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)

    if bias is not None:
        if len(bias) == 2:
            if bias.shape[0] != 1:
                raise RuntimeError(f"bias must be a 1D tensor")
            bias = bias[0]
        output_tensor += bias

    # First check if there is a fused activation in the program config
    if program_config is not None and hasattr(program_config, "fused_activation") and program_config.fused_activation:
        program_config_activation = program_config.fused_activation.op_type
        output_tensor = get_golden_function_for_activation(program_config_activation)(output_tensor)

    # Do the composite op activation function if it is requested
    elif activation is not None:
        output_tensor = get_golden_function_for_activation(activation)(output_tensor)

    while len(output_tensor.shape) > len(input_tensor_a.shape):
        output_tensor = output_tensor.squeeze(0)
    return output_tensor


ttnn.attach_golden_function(
    ttnn.linear,
    golden_function=_golden_function,
)


def _golden_function(input_tensor, mat1_tensor, mat2_tensor, alpha=1.0, beta=1.0, out_tensor=None, **kwargs):
    import torch

    if beta == 0:
        # TTNN intentionally ignores the addend when beta is zero, including an otherwise invalid addend shape.
        result = alpha * torch.matmul(mat1_tensor, mat2_tensor)
        if out_tensor is not None:
            out_tensor.copy_(result)
            return out_tensor
        return result
    return torch.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha=alpha, beta=beta, out=out_tensor)


ttnn.attach_golden_function(
    ttnn.addmm,
    golden_function=_golden_function,
)


def _golden_function_matmul_batched_weights(input_tensor_a, input_tensors_b, *_, **__):
    import torch

    # One input tensor a multiplied against each of the batched weight tensors b.
    return [torch.matmul(input_tensor_a, b.to(input_tensor_a.dtype)) for b in input_tensors_b]


ttnn.attach_golden_function(ttnn.matmul_batched_weights, golden_function=_golden_function_matmul_batched_weights)


def _golden_function_sparse_matmul(
    input_tensor_a,
    input_tensor_b,
    *,
    sparsity,
    is_input_a_sparse=False,
    is_input_b_sparse=True,
    nnz=None,
    indices=None,
    optional_output_tensor=None,
    **__,
):
    import torch

    a, b = input_tensor_a, input_tensor_b
    if indices is not None:
        raise NotImplementedError("sparse_matmul golden does not support indexed/gather output")
    if not is_input_a_sparse and not is_input_b_sparse:
        raise ValueError("sparse_matmul requires at least one sparse input")

    if is_input_a_sparse:
        dense = torch.matmul(a, b.to(a.dtype))
        mask = (sparsity != 0).reshape(dense.shape[:-2])
        expanded_output = dense * mask.unsqueeze(-1).unsqueeze(-1).to(dense.dtype)
    else:
        # Dense-A/sparse-B mode forms every pair from A's batch dims and B's sparse-group dims.
        k = a.shape[-1]
        a_batch, b_batch = a.shape[:-2], b.shape[:-2]
        a_exp = a.reshape(*a_batch, *([1] * len(b_batch)), a.shape[-2], k)
        b_exp = b.reshape(*([1] * len(a_batch)), *b_batch, k, b.shape[-1])
        dense = torch.matmul(a_exp, b_exp.to(a.dtype))
        mask = (sparsity != 0).reshape(*a_batch, *b_batch)
        expanded_output = dense * mask.unsqueeze(-1).unsqueeze(-1).to(dense.dtype)

    if optional_output_tensor is not None and optional_output_tensor.shape != expanded_output.shape:
        raise NotImplementedError("sparse_matmul golden does not support compact nnz output")
    return expanded_output


ttnn.attach_golden_function(ttnn.sparse_matmul, golden_function=_golden_function_sparse_matmul)


ttnn.Tensor.__matmul__ = lambda self, *args, **kwargs: ttnn.matmul(self, *args, **kwargs)


__all__ = []
