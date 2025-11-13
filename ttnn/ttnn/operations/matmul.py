# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import ttnn
from ttnn.decorators import get_golden_function

MatmulProgramConfig = ttnn._ttnn.operations.matmul.MatmulProgramConfig
MatmulMultiCoreReuseProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseProgramConfig
MatmulMultiCoreReuseMultiCastProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastProgramConfig
MatmulMultiCoreReuseMultiCast1DProgramConfig = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCast1DProgramConfig
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig = (
    ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
)


def _get_golden_activation_function(activation):
    import torch

    golden_activations_map = {
        ttnn.UnaryOpType.RELU: torch.nn.functional.relu,
        ttnn.UnaryOpType.RELU6: torch.nn.functional.relu6,
        ttnn.UnaryOpType.SILU: torch.nn.functional.silu,
        ttnn.UnaryOpType.MISH: torch.nn.functional.mish,
        ttnn.UnaryOpType.SIGMOID: torch.nn.functional.sigmoid,
        ttnn.UnaryOpType.TANH: torch.nn.functional.tanh,
        ttnn.UnaryOpType.LOG: torch.log,
        ttnn.UnaryOpType.SOFTPLUS: torch.nn.functional.softplus,
        ttnn.UnaryOpType.GELU: torch.nn.functional.gelu,
        ttnn.UnaryOpType.SQRT: torch.sqrt,
    }

    if activation in golden_activations_map:
        return golden_activations_map[activation]
    else:
        raise RuntimeError(f"{activation} is not supported as activation function")


# Given a string like "relu6", return the corresponding torch activation function, for example torch.nn.functional.relu6
def _get_torch_activation_from_string(name: str):
    # Remove "_approx" suffix if present in the activation name
    if name.endswith("_approx"):
        name = name[:-7]

    op = getattr(ttnn, name, None)
    if op is None:
        raise RuntimeError(f"Unsupported activation: {name}")
    return get_golden_function(op)


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
        output_tensor = _get_golden_activation_function(program_config_activation)(output_tensor)

    # Do the composite op activation function if it is requested
    elif activation is not None:
        output_tensor = _get_torch_activation_from_string(activation)(output_tensor)

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
        output_tensor = _get_golden_activation_function(program_config_activation)(output_tensor)

    # Do the composite op activation function if it is requested
    elif activation is not None:
        output_tensor = _get_torch_activation_from_string(activation)(output_tensor)

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
