# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn
from ttnn.decorators import get_golden_function


def _get_golden_map_for_unary_op():
    import torch

    act_map = {
        ttnn.UnaryOpType.RELU: torch.nn.functional.relu,
        ttnn.UnaryOpType.RELU6: torch.nn.functional.relu6,
        ttnn.UnaryOpType.SILU: torch.nn.functional.silu,
        ttnn.UnaryOpType.MISH: torch.nn.functional.mish,
        ttnn.UnaryOpType.SIGMOID: torch.nn.functional.sigmoid,
        ttnn.UnaryOpType.HARDSIGMOID: torch.nn.functional.hardsigmoid,
        ttnn.UnaryOpType.TANH: torch.nn.functional.tanh,
        ttnn.UnaryOpType.LOG: torch.log,
        ttnn.UnaryOpType.SOFTPLUS: torch.nn.functional.softplus,
        ttnn.UnaryOpType.GELU: torch.nn.functional.gelu,
        ttnn.UnaryOpType.SQRT: torch.sqrt,
    }

    return act_map


def _get_golden_function_for_unary_op_type(activation: ttnn.UnaryOpType) -> callable:
    """
    Return a torch golden function for a UnaryOpType.
    e.g. ttnn.UnaryOpType.RELU -> torch.nn.functional.relu
    """
    if not isinstance(activation, ttnn.UnaryOpType):
        raise RuntimeError(f"{activation} is not a UnaryOpType")

    act_map = _get_golden_map_for_unary_op()

    if activation in act_map:
        return act_map[activation]

    raise RuntimeError(f"{activation} is not supported as activation function")


def _get_golden_activation_from_string(activation: str) -> callable:
    """Return a torch golden function for a string activation.
    e.g. "relu6" -> torch.nn.functional.relu6.
    """
    name = activation[:-7] if activation.endswith("_approx") else activation
    op = getattr(ttnn, name, None)
    if op is None:
        raise RuntimeError(f"Unsupported activation: {activation}")
    return get_golden_function(op)


def get_golden_function_for_activation(activation: Optional[object]) -> Optional[callable]:
    """Return a torch golden function for an activation.

    The activation can be:
    - string name (e.g., 'relu6', 'gelu_approx')
    - ttnn.UnaryWithParam
    - ttnn.UnaryOpType
    - None (will return None)
    """
    if activation is None:
        return None

    # String Inputs
    if isinstance(activation, str):
        return _get_golden_activation_from_string(activation)

    # UnaryWithParam -> just extract its op_type
    if hasattr(activation, "op_type"):
        activation = activation.op_type

    # Expect to be a UnaryOpType
    if not isinstance(activation, ttnn.UnaryOpType):
        raise RuntimeError(f"{activation} is not supported as activation function")

    return _get_golden_function_for_unary_op_type(activation)
