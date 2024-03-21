# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import math

from typing import Union

import tt_lib as ttl

import ttnn

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_math_op_function_unary(name, ttl_math_op_function, op_name):
    def _torch_math(input_tensor: ttnn.Tensor, **_):
        name_to_torch_function = {
            "i0": torch.i0,
            "isfinite": torch.isfinite,
            "isinf": torch.inf,
            "isnan": torch.isnan,
            "isneginf": torch.isneginf,
            "isposinf": torch.isposinf,
            "lgamma": torch.lgamma,
            "log10": torch.log10,
            "log1p": torch.log1p,
            "log2": torch.log2,
            "multigammaln": torch_multigammaln,
            "neg": torch.neg,
            "abs": torch.abs,
            "cbrt": torch_cbrt,
            "deg2rad": torch.deg2rad,
            "digamma": torch.digamma,
            "erf": torch.erf,
            "erfc": torch.erfc,
            "erfinv": torch.erfinv,
            "exp2": torch.exp2,
            "expm1": torch.expm1,
            "rad2deg": torch.rad2deg,
            "reciprocal": torch.reciprocal,
            "sqrt": torch.sqrt,
            "square": torch.square,
            "tril": torch.tril,
            "triu": torch.triu,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _math_op_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
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
        validate_input_tensors=_math_op_validate_input_tensors,
        torch_function=_torch_math,
    )
    def math_op_function(
        input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_math_op_function(input_tensor, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    math_op_function.__name__ = f"ttnn.{(name)}"
    math_op_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies the {op_name} function to the elements of the input tensor :attr:`input_tensor`.

        .. math::
            {(op_name)}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{(name)}(tensor)

        """
    setattr(THIS_MODULE, name, math_op_function)


TTL_MATH_OP_FUNCTIONS_UNARY = [
    ("i0", ttl.tensor.i0, "i0"),
    ("isfinite", ttl.tensor.isfinite, "isfinite"),
    ("isinf", ttl.tensor.isinf, "isinf"),
    ("isnan", ttl.tensor.isnan, "isnan"),
    ("isneginf", ttl.tensor.isneginf, "isneginf"),
    ("isposinf", ttl.tensor.isposinf, "isposinf"),
    ("lgamma", ttl.tensor.lgamma, "lgamma"),
    ("log10", ttl.tensor.log10, "log10"),
    ("log1p", ttl.tensor.log1p, "log1p"),
    ("log2", ttl.tensor.log2, "log2"),
    ("multigammaln", ttl.tensor.multigammaln, "multigammaln"),
    ("neg", ttl.tensor.neg, "neg"),
    ("abs", ttl.tensor.abs, "abs"),
    ("cbrt", ttl.tensor.cbrt, "cbrt"),
    ("deg2rad", ttl.tensor.deg2rad, "deg2rad"),
    ("digamma", ttl.tensor.digamma, "digamma"),
    ("erf", ttl.tensor.erf, "erf"),
    ("erfc", ttl.tensor.erfc, "erfc"),
    ("erfinv", ttl.tensor.erfinv, "erfinv"),
    ("exp2", ttl.tensor.exp2, "exp2"),
    ("expm1", ttl.tensor.expm1, "expm1"),
    ("rad2deg", ttl.tensor.rad2deg, "rad2deg"),
    ("reciprocal", ttl.tensor.recip, "reciprocal"),
    ("sqrt", ttl.tensor.sqrt, "sqrt"),
    ("square", ttl.tensor.square, "square"),
    ("tril", ttl.tensor.tril, "tril"),
    ("triu", ttl.tensor.triu, "triu"),
]


for math_op_function_name, ttl_math_op_function, op_name in TTL_MATH_OP_FUNCTIONS_UNARY:
    register_ttl_math_op_function_unary(math_op_function_name, ttl_math_op_function, op_name)


def torch_cbrt(x, *args, **kwargs):
    return torch.sgn(x) * torch.pow(torch.abs(x), 1.0 / 3)


def torch_multigammaln(x, *args, **kwargs):
    result = torch.lgamma(x)
    result += torch.lgamma(x - 0.5)
    result += torch.lgamma(x - 1.0)
    result += torch.lgamma(x - 1.5)
    result += 3.434189657547
    return result


def register_ttl_math_binary_function(name, ttl_math_binary_function, op_name):
    def _torch_math_binary(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, **_):
        name_to_torch_function = {
            "atan2": torch.atan2,
            "hypot": torch.hypot,
            "squared_difference": torch_squared_difference,
        }

        input_shape_a = input_tensor_a.shape
        slices = [slice(0, dim) for dim in input_shape_a]
        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)
        input_tensor_a = input_tensor_a[slices]

        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

        torch_function = name_to_torch_function[name]
        return torch_function(input_tensor_a, input_tensor_b)

    def _math_binary_validate_input_tensors(operation_name, input_tensor_a, input_tensor_b, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_a,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_b,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_math_binary_validate_input_tensors,
        torch_function=_torch_math_binary,
    )
    def math_binary_function(
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: Union[ttnn.Tensor, int, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        if not isinstance(input_tensor_a, ttnn.Tensor) or not isinstance(input_tensor_b, ttnn.Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor_a) or not ttnn.is_tensor_storage_on_device(input_tensor_b):
            raise RuntimeError("input_tensors must be on device!")

        original_shape = input_tensor_a.shape

        input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

        output_tensor = ttl_math_binary_function(input_tensor_a, input_tensor_b, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    math_binary_function.__name__ = f"ttnn.{name}"
    math_binary_function.decorated_function.__doc__ = f"""{name}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Performs eltwise-binary {op_name} operation on two tensors :attr:`input_a` and :attr:`input_b`.

        .. math::
            {name.replace('_',' ')}(\\mathrm{{input\\_tensor\\_a}}_i \\; , \\; \\mathrm{{input\\_tensor\\_b}}_i  \\; \\; or \\; \\; \\mathrm{{scalar}})

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`

        Example::
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 1], [4, 4]]), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor1, tensor2)
        """

    setattr(THIS_MODULE, name, math_binary_function)


TTL_BINARY_MATH_FUNCTIONS = [
    ("atan2", ttl.tensor.atan2, "atan2"),
    ("hypot", ttl.tensor.hypot, "hypotenuse"),
    ("squared_difference", ttl.tensor.squared_difference, "squared_difference (input_a - input_b)^2"),
]


for math_binary_function_name, ttl_math_binary_function, op_name in TTL_BINARY_MATH_FUNCTIONS:
    register_ttl_math_binary_function(math_binary_function_name, ttl_math_binary_function, op_name)


def torch_squared_difference(x, y, *args, **kwargs):
    return torch.square(torch.sub(x, y))


def register_ttl_lerp_function(name, ttl_lerp_function, op_name):
    def _is_scalar(value):
        return isinstance(value, (int, float))

    def _torch_math_binary(
        input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, weight: Union[ttnn.Tensor, int, float], **_
    ):
        name_to_torch_function = {
            "lerp": torch.lerp,
        }

        input_shape_a = input_tensor_a.shape
        slices = [slice(0, dim) for dim in input_shape_a]
        input_tensor_a = ttnn.from_device(input_tensor_a)
        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_a = ttnn.to_torch(input_tensor_a)
        input_tensor_a = input_tensor_a[slices]

        input_shape_b = input_tensor_b.shape
        slices = [slice(0, dim) for dim in input_shape_b]
        input_tensor_b = ttnn.from_device(input_tensor_b)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_b = ttnn.to_torch(input_tensor_b)
        input_tensor_b = input_tensor_b[slices]

        if not _is_scalar(weight):
            weight_shape = weight.shape
            slices = [slice(0, dim) for dim in weight_shape]
            weight = ttnn.from_device(weight)
            weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
            weight = ttnn.to_torch(weight)
            weight = weight[slices]

        torch_function = name_to_torch_function[name]
        return torch_function(input_tensor_a, input_tensor_b, weight)

    def _math_binary_validate_input_tensors(operation_name, input_tensor_a, input_tensor_b, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_a,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor_b,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_math_binary_validate_input_tensors,
        torch_function=_torch_math_binary,
    )
    def lerp_function(
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        weight: Union[ttnn.Tensor, int, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        if not (input_tensor_a.shape == input_tensor_b.shape):
            raise RuntimeError("input_tensors must be of same size!")

        if not isinstance(input_tensor_a, ttnn.Tensor) or not isinstance(input_tensor_b, ttnn.Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor_a) or not ttnn.is_tensor_storage_on_device(input_tensor_b):
            raise RuntimeError("input_tensors must be on device!")

        if isinstance(weight, ttnn.Tensor) and not (input_tensor_a.shape == weight.shape):
            raise RuntimeError("weight tensor must be of same size!")

        if isinstance(weight, ttnn.Tensor) and not ttnn.is_tensor_storage_on_device(weight):
            raise RuntimeError("weight tensor must be on device!")

        original_shape = input_tensor_a.shape

        input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

        if isinstance(weight, ttnn.Tensor):
            weight = ttnn.unsqueeze_to_4D(weight)

        output_tensor = ttl_lerp_function(input_tensor_a, input_tensor_b, weight, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    lerp_function.__name__ = f"ttnn.{name}"
    lerp_function.decorated_function.__doc__ = f"""{name}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, weight: Union[ttnn.Tensor, int, float], *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Performs eltwise-binary {op_name} operation on two tensors :attr:`input_a` and :attr:`input_b`, based on :attr:`weight`.

        .. math::
            {name.replace('_',' ')}(\\mathrm{{input\\_tensor\\_a}}_i \\; , \\mathrm{{input\\_tensor\\_b}}_i \\; , \\; \\mathrm{{weight_tensor}}_i  \\; \\; or \\; \\; \\mathrm{{weight_scalar}})

        Args:
            * :attr:`input_tensor_a`
            * :attr:`input_tensor_b`
            * :attr:`weight`

        Example::
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 1], [4, 4]]), dtype=torch.bfloat16)), device)
            >>> weight = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 1], [4, 4]]), dtype=torch.bfloat16)), device)
            >>> output = ttnn.{name}(tensor1, tensor2, weight)
        """

    setattr(THIS_MODULE, name, lerp_function)


TTL_LERP_FUNCTION = [
    ("lerp", ttl.tensor.lerp, "linear interpolation"),
]


for lerp_function_name, ttl_lerp_function, op_name in TTL_LERP_FUNCTION:
    register_ttl_lerp_function(lerp_function_name, ttl_lerp_function, op_name)


def _is_scalar(value):
    return isinstance(value, (int, float))


def register_ttl_math_unary_function_with_float(name, ttl_math_unary_function, op_name, param):
    def _torch_math_unary(input_tensor: ttnn.Tensor, parameter, **_):
        import torch

        name_to_torch_function = {
            "polygamma": torch_polygamma,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        if name == "polygamma":
            return torch_function(input_tensor, scalar=parameter)
        else:
            return torch_function(input_tensor, parameter)

    def _math_unary_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
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
        validate_input_tensors=_math_unary_validate_input_tensors,
        torch_function=_torch_math_unary,
    )
    def math_unary_function(
        input_tensor: ttnn.Tensor, parameter: float, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not _is_scalar(parameter):
            raise TypeError("Expected second argument to be a scalar")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_math_unary_function(input_tensor, parameter, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    math_unary_function.__name__ = f"ttnn.{(name)}"
    math_unary_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

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
    setattr(THIS_MODULE, name, math_unary_function)


TTL_MATH_UNARY_FUNCTIONS_WITH_FLOAT_PARAM = [
    ("polygamma", ttl.tensor.polygamma, "polygamma", "n"),
]

for math_unary_function_name, ttl_math_unary_function, name, param in TTL_MATH_UNARY_FUNCTIONS_WITH_FLOAT_PARAM:
    register_ttl_math_unary_function_with_float(math_unary_function_name, ttl_math_unary_function, name, param)


def torch_polygamma(x, *args, **kwargs):
    n = kwargs.pop("scalar")
    return torch.special.polygamma(n, x)


__all__ = []
