# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_unary_function(name, ttl_unary_function):
    def _torch_unary(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_torch_function = {
            "exp": torch.exp,
            "tanh": torch.tanh,
            "gelu": torch.nn.functional.gelu,
            "rsqrt": torch.rsqrt,
            "relu": torch.relu,
            "silu": torch.nn.functional.silu,
            "log": torch.log,
            "sin": torch.sin,
            "cos": torch.cos,
            "tan": torch.tan,
            "asin": torch.asin,
            "acos": torch.acos,
            "atan": torch.atan,
            "sinh": torch.sinh,
            "cosh": torch.cosh,
            "asinh": torch.asinh,
            "acosh": torch.acosh,
            "atanh": torch.atanh,
            "logical_not": torch.logical_not,
            "signbit": torch.signbit,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _unary_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_unary_validate_input_tensors,
        torch_function=_torch_unary,
    )
    def unary_function(
        input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected first argument to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_unary_function(input_tensor, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    unary_function.__name__ = f"ttnn.{name}"
    unary_function.decorated_function.__doc__ = f"""{name}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Applies {name} to :attr:`input_tensor` element-wise.

        .. math::
            {name.replace('_',' ')}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor)

        {unary_function.__doc__}

        """
    setattr(THIS_MODULE, name, unary_function)


TTL_UNARY_FUNCTIONS = [
    ("exp", ttl.tensor.exp),
    ("tanh", ttl.tensor.tanh),
    ("gelu", ttl.tensor.gelu),
    ("relu", ttl.tensor.relu),
    ("rsqrt", ttl.tensor.rsqrt),
    ("silu", ttl.tensor.silu),
    ("log", ttl.tensor.log),
    ("sin", ttl.tensor.sin),
    ("cos", ttl.tensor.cos),
    ("tan", ttl.tensor.tan),
    ("asin", ttl.tensor.asin),
    ("acos", ttl.tensor.acos),
    ("atan", ttl.tensor.atan),
    ("sinh", ttl.tensor.sinh),
    ("cosh", ttl.tensor.cosh),
    ("asinh", ttl.tensor.asinh),
    ("acosh", ttl.tensor.acosh),
    ("atanh", ttl.tensor.atanh),
    ("logical_not", ttl.tensor.logical_not_unary),
    ("signbit", ttl.tensor.signbit),
]


for unary_function_name, ttl_unary_function in TTL_UNARY_FUNCTIONS:
    register_ttl_unary_function(unary_function_name, ttl_unary_function)


def _is_scalar(value):
    return isinstance(value, (int, float))


def register_ttl_unary_function_with_float(name, ttl_unary_function, op_name, param):
    def _torch_unary(input_tensor: ttnn.Tensor, parameter, **_):
        import torch

        name_to_torch_function = {
            "logit": torch.logit,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        return torch_function(input_tensor, parameter)

    def _unary_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
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
        validate_input_tensors=_unary_validate_input_tensors,
        torch_function=_torch_unary,
    )
    def unary_function(
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

        output_tensor = ttl_unary_function(input_tensor, parameter, output_mem_config=memory_config)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    unary_function.__name__ = f"ttnn.{(name)}"
    unary_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

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
    setattr(THIS_MODULE, name, unary_function)


TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAM = [
    ("logit", ttl.tensor.logit, "logit", "eps"),
]

for unary_function_name, ttl_unary_function, name, param in TTL_UNARY_FUNCTIONS_WITH_FLOAT_PARAM:
    register_ttl_unary_function_with_float(unary_function_name, ttl_unary_function, name, param)


__all__ = []
