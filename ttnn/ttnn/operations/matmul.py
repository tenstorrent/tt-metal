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
MatmulMultiCoreReuseOptimizedProgramFactory = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseOptimizedProgramFactory
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

    return torch.addmm(input_tensor, mat1_tensor, mat2_tensor, alpha=alpha, beta=beta, out=out_tensor)


ttnn.attach_golden_function(
    ttnn.addmm,
    golden_function=_golden_function,
)


def _golden_function_matmul(
    input_tensor_a,
    input_tensor_b,
    transpose_a=False,
    transpose_b=False,
    *,
    program_config=None,
    activation=None,
    dtype=None,
):
    import torch

    if transpose_a:
        input_tensor_a = input_tensor_a.transpose(-1, -2)
    if transpose_b:
        input_tensor_b = input_tensor_b.transpose(-1, -2)

    output = input_tensor_a @ input_tensor_b.to(input_tensor_a.dtype)
    if program_config is not None and getattr(program_config, "fused_activation", None):
        output = get_golden_function_for_activation(program_config.fused_activation.op_type)(output)
    elif activation is not None:
        output = get_golden_function_for_activation(activation)(output)
    if dtype == ttnn.float32:
        output = output.to(torch.float32)
    elif dtype == ttnn.bfloat16:
        output = output.to(torch.bfloat16)
    return output


def _golden_function_matmul_batched_weights(
    input_tensor_a,
    input_tensors_b,
    transpose_a=False,
    transpose_b=False,
    *,
    program_config=None,
    activation=None,
    dtype=None,
    **kwargs,
):
    return [
        _golden_function_matmul(
            input_tensor_a,
            input_tensor_b,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            program_config=program_config,
            activation=activation,
            dtype=dtype,
        )
        for input_tensor_b in input_tensors_b
    ]


ttnn.attach_golden_function(
    ttnn.matmul_batched_weights,
    golden_function=_golden_function_matmul_batched_weights,
)


def _golden_function_sparse_matmul(
    input_tensor_a,
    input_tensor_b,
    *,
    sparsity,
    is_input_a_sparse=False,
    is_input_b_sparse=True,
    dtype=None,
    optional_output_tensor=None,
    **kwargs,
):
    import torch

    if not is_input_a_sparse and not is_input_b_sparse:
        raise ValueError("sparse_matmul requires at least one sparse input")

    if is_input_a_sparse:
        # Sparse A modes pair matching expert coordinates from A and B.
        output = torch.matmul(input_tensor_a, input_tensor_b)
    else:
        # Dense A batches are crossed with every sparse expert in B.  This intentionally
        # creates [...A batches..., ...B batches..., M, N], matching the device op.
        a_batch_shape = input_tensor_a.shape[:-2]
        b_batch_shape = input_tensor_b.shape[:-2]
        a = input_tensor_a.reshape(*a_batch_shape, *([1] * len(b_batch_shape)), *input_tensor_a.shape[-2:])
        b = input_tensor_b.reshape(*([1] * len(a_batch_shape)), *b_batch_shape, *input_tensor_b.shape[-2:])
        output = torch.matmul(a, b)

    mask = (sparsity != 0).reshape(output.shape[:-2])
    output = output * mask[..., None, None].to(output.dtype)
    if dtype == ttnn.float32:
        output = output.to(torch.float32)
    elif dtype == ttnn.bfloat16:
        output = output.to(torch.bfloat16)
    if optional_output_tensor is not None:
        if dtype is None:
            output = output.to(optional_output_tensor.dtype)
        if optional_output_tensor.shape != output.shape or optional_output_tensor.dtype != output.dtype:
            raise ValueError("sparse_matmul optional_output_tensor must match the result shape and dtype")
        optional_output_tensor.copy_(output)
        return optional_output_tensor
    return output


ttnn.attach_golden_function(
    ttnn.sparse_matmul,
    golden_function=_golden_function_sparse_matmul,
)


ttnn.Tensor.__matmul__ = lambda self, *args, **kwargs: ttnn.matmul(self, *args, **kwargs)


__all__ = []
