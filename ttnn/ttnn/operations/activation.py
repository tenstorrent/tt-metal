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
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_activation_function(ttl_input_tensor, output_mem_config=memory_config)

        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    activation_function.__name__ = f"ttnn.{(name)}"
    activation_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

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


def torch_relu_max(x, *args, **kwargs):
    upper_limit = kwargs.pop("scalar")
    capped_tensor = torch.min(x, torch.tensor(upper_limit, dtype=x.dtype))
    return torch.relu(capped_tensor)


def torch_relu_min(x, *args, **kwargs):
    lower_limit = kwargs.pop("scalar")
    capped_tensor = torch.max(x, torch.tensor(lower_limit, dtype=x.dtype))
    return torch.relu(capped_tensor)


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
            "relu_max": torch_relu_max,
            "relu_min": torch_relu_min,
            "softshrink": F.softshrink,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        if name == "relu_max" or name == "relu_min" or name == "heaviside" or name == "prelu":
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
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter):
            raise TypeError("Expected second argument to be a float")

        if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_activation_function(ttl_input_tensor, parameter, output_mem_config=memory_config)

        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    activation_function.__name__ = f"ttnn.{(name)}"
    activation_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

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
        name_to_torch_function = {
            "clip": torch.clamp,
        }
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
        ttl_input_tensor = input_tensor.value

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter1) or not _is_scalar(parameter2):
            raise TypeError("Expected parameters to be a float")

        if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
            raise RuntimeError("input_tensor must be on device!")
        ttl_input_tensor = input_tensor.value

        ttl_output_tensor = ttl_activation_function(
            ttl_input_tensor, parameter1, parameter2, output_mem_config=memory_config
        )

        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    activation_function.__name__ = f"ttnn.{(name)}"
    activation_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

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
    ("softplus", ttl.tensor.softplus, "softplus"),
]

TTL_ACTIVATION_FUNCTIONS_WITH_FLOAT_PARAM = [
    ("hardshrink", ttl.tensor.hardshrink, "hardshrink", "lambda"),
    ("heaviside", ttl.tensor.heaviside, "heaviside", "value"),
    ("leaky_relu", ttl.tensor.leaky_relu, "leaky relu", "slope"),
    ("prelu", ttl.tensor.prelu, "prelu", "weight"),
    ("elu", ttl.tensor.elu, "elu", "alpha"),
    ("relu_max", ttl.tensor.relu_max, "relu max", "upper-limit"),
    ("relu_min", ttl.tensor.relu_min, "relu min", "lower-limit"),
    ("softshrink", ttl.tensor.softshrink, "softshrink", "lambda"),
]

TTL_ACTIVATION_FUNCTIONS_WITH_TWO_FLOAT_PARAM = [
    ("clip", ttl.tensor.clip, "Clip", "min", "max"),
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

__all__ = []
