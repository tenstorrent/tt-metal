# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import ttnn
from ttnn.decorators import get_golden_function
from ttnn.operations.golden_common import golden_apply_fused_activations

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

    output_tensor = golden_apply_fused_activations(output_tensor, activation, program_config=program_config)

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

    output_tensor = golden_apply_fused_activations(output_tensor, activation, program_config=program_config)

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

ttnn.Tensor.__matmul__ = lambda self, *args, **kwargs: ttnn.matmul(self, *args, **kwargs)


__all__ = []
