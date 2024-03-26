# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import math

from typing import Union

import tt_lib as ttl

import ttnn

import torch.nn.functional as F

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_activation_function_unary(name, ttl_activation_function, op_name):
    def _torch_activation(input_tensor: ttnn.Tensor, **_):
        name_to_torch_function = {
            "hardsigmoid": F.hardsigmoid,
            "hardswish": F.hardswish,
            "hardtanh": F.hardtanh,
            "log_sigmoid": F.logsigmoid,
            "mish": lambda _x: F.mish(_x.to(torch.float)),
            "relu6": F.relu6,
            "sigmoid": torch.sigmoid,
            "sign": torch.sign,
            "softsign": F.softsign,
            "swish": F.hardswish,
            "softplus": F.softplus,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _activation_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_activation_validate_input_tensors,
        torch_function=_torch_activation,
    )
    def activation_function(
        input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    activation_function.__name__ = f"ttnn.{(name)}"
    activation_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies the {op_name} function to the elements of the input tensor :attr:`input_tensor`.

        .. math::
            {(op_name)}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{(name)}(tensor)

        """
    setattr(THIS_MODULE, name, activation_function)


def _is_scalar(value):
    return isinstance(value, (int, float))


def torch_heaviside(x, *args, **kwargs):
    value = kwargs.pop("scalar")
    result = torch.heaviside(x, torch.tensor(value, dtype=x.dtype))
    return result


def torch_prelu(x, *args, **kwargs):
    weight = kwargs.pop("scalar")
    result = F.prelu(x, torch.tensor(weight, dtype=x.dtype))
    return result


def register_ttl_activation_function_with_float(name, ttl_activation_function, op_name, param):
    def _torch_activation(input_tensor: ttnn.Tensor, parameter, **_):
        name_to_torch_function = {
            "hardshrink": F.hardshrink,
            "heaviside": torch_heaviside,
            "leaky_relu": F.leaky_relu,
            "prelu": torch_prelu,
            "elu": F.elu,
            "softshrink": F.softshrink,
            "tanhshrink": F.tanhshrink,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        if name == "heaviside" or name == "prelu":
            return torch_function(input_tensor, scalar=parameter)
        else:
            return torch_function(input_tensor, parameter)

    def _activation_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_activation_validate_input_tensors,
        torch_function=_torch_activation,
    )
    def activation_function(
        input_tensor: ttnn.Tensor, parameter: float, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter):
            raise TypeError("Expected second argument to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, parameter, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    activation_function.__name__ = f"ttnn.{(name)}"
    activation_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies the {op_name} function to the elements of the input tensor :attr:`input_tensor` with :attr:`{param}` parameter.

        .. math::
            {(op_name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param})

        Args:
            * :attr:`input_tensor`
            * :attr:`{param}`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{(name)}(tensor, {param})

        """
    setattr(THIS_MODULE, name, activation_function)


def register_ttl_activation_function_with_two_float(name, ttl_activation_function, op_name, param1_name, param2_name):
    def _torch_activation(input_tensor: ttnn.Tensor, parameter1, parameter2, **_):
        name_to_torch_function = {"clip": torch.clamp, "threshold": F.threshold, "softplus": F.softplus}
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor, parameter1, parameter2)

    def _activation_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_activation_validate_input_tensors,
        torch_function=_torch_activation,
    )
    def activation_function(
        input_tensor: ttnn.Tensor,
        parameter1: float,
        parameter2: float,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter1) or not _is_scalar(parameter2):
            raise TypeError("Expected parameters to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, parameter1, parameter2, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    activation_function.__name__ = f"ttnn.{(name)}"
    activation_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies the {op_name} function to the elements of the input tensor :attr:`input_tensor` with :attr:`{param1_name}` and :attr:`{param2_name}`  parameters.

        .. math::
            {(name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param1_name} \\; , \\; {param2_name})

        Args:
            * :attr:`input_tensor`
            * :attr:`{param1_name}`
            * :attr:`{param2_name}`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{(name)}(tensor, {param1_name}, {param2_name})

        """
    setattr(THIS_MODULE, name, activation_function)


def torch_reglu(input_tensor, *args, **kwargs):
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * F.relu(tensB)


def torch_swiglu(input_tensor, *args, **kwargs):
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * F.silu(tensB)


def torch_geglu(input_tensor, *args, **kwargs):
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * F.gelu(tensB)


def register_ttl_activation_function_glu(name, ttl_activation_function, op_name, param):
    def _torch_activation(input_tensor: ttnn.Tensor, dim: int = -1, **_):
        name_to_torch_function = {
            "glu": F.glu,
            "reglu": torch_reglu,
            "swiglu": torch_swiglu,
            "geglu": torch_geglu,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        return torch_function(input_tensor, dim=dim)

    def _activation_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_activation_validate_input_tensors,
        torch_function=_torch_activation,
    )
    def activation_function(
        input_tensor: ttnn.Tensor, dim: int = -1, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        input_shape = tuple(input_tensor.shape)
        last_dim = input_shape[-1]
        glu_shape = input_shape[:-1] + (int(last_dim / 2),)

        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(dim):
            raise TypeError("Expected second argument to be a float")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_activation_function(input_tensor, dim, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, ttnn.Shape(glu_shape))
        return output_tensor

    activation_function.__name__ = f"ttnn.{(name)}"
    activation_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, dim: int = -1, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies the {op_name} function to the elements of the input tensor :attr:`input_tensor` split along :attr:`{param}`.

        .. math::
            {(name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param})

        Args:
            * :attr:`input_tensor`
            * :attr:`{param}`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((32, 64), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{(name)}(tensor, {param})

        """
    setattr(THIS_MODULE, name, activation_function)


TTL_ACTIVATION_FUNCTIONS_UNARY = [
    ("hardsigmoid", ttl.tensor.hardsigmoid, "hardsigmoid"),
    ("hardswish", ttl.tensor.hardswish, "hardswish"),
    ("hardtanh", ttl.tensor.hardtanh, "hardtanh"),
    ("log_sigmoid", ttl.tensor.log_sigmoid, "log sigmoid"),
    ("mish", ttl.tensor.mish, "mish"),
    ("relu6", ttl.tensor.relu6, "relu6"),
    ("sigmoid", ttl.tensor.sigmoid, "sigmoid"),
    ("sign", ttl.tensor.sign, "sign"),
    ("softsign", ttl.tensor.softsign, "softsign"),
    ("swish", ttl.tensor.swish, "swish"),
    ("tanhshrink", ttl.tensor.tanhshrink, "tanhshrink"),
]

TTL_ACTIVATION_FUNCTIONS_WITH_FLOAT_PARAM = [
    ("hardshrink", ttl.tensor.hardshrink, "hardshrink", "lambda"),
    ("heaviside", ttl.tensor.heaviside, "heaviside", "value"),
    ("leaky_relu", ttl.tensor.leaky_relu, "leaky relu", "slope"),
    ("prelu", ttl.tensor.prelu, "prelu", "weight"),
    ("elu", ttl.tensor.elu, "elu", "alpha"),
    ("softshrink", ttl.tensor.softshrink, "softshrink", "lambda"),
]

TTL_ACTIVATION_FUNCTIONS_WITH_TWO_FLOAT_PARAM = [
    ("clip", ttl.tensor.clip, "clip", "min", "max"),
    ("threshold", ttl.tensor.threshold, "threshold", "value", "threshold"),
    ("softplus", ttl.tensor.softplus, "softplus", "beta", "threshold"),
]

TTL_ACTIVATION_FUNCTIONS_GLU = [
    ("glu", ttl.tensor.glu, "Gated Linear Units (GLU)", "dim"),
    ("reglu", ttl.tensor.reglu, "Rectified Gated Linear Units (ReGLU)", "dim"),
    ("swiglu", ttl.tensor.swiglu, "Swish Gated Linear Units (SwiGLU)", "dim"),
    ("geglu", ttl.tensor.geglu, "Gaussian Error Gated Linear Units (GeGLU)", "dim"),
]

for activation_function_name, ttl_activation_function, name, param in TTL_ACTIVATION_FUNCTIONS_WITH_FLOAT_PARAM:
    register_ttl_activation_function_with_float(activation_function_name, ttl_activation_function, name, param)

for (
    activation_function_name,
    ttl_activation_function,
    name,
    param1,
    param2,
) in TTL_ACTIVATION_FUNCTIONS_WITH_TWO_FLOAT_PARAM:
    register_ttl_activation_function_with_two_float(
        activation_function_name, ttl_activation_function, name, param1, param2
    )

for activation_function_name, ttl_activation_function, name in TTL_ACTIVATION_FUNCTIONS_UNARY:
    register_ttl_activation_function_unary(activation_function_name, ttl_activation_function, name)

for activation_function_name, ttl_activation_function, op_name, param in TTL_ACTIVATION_FUNCTIONS_GLU:
    register_ttl_activation_function_glu(activation_function_name, ttl_activation_function, op_name, param)

__all__ = []
