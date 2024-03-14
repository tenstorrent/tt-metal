# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

import ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []


def torch_mac(input, tensor1, tensor2):
    import torch

    return torch.add(torch.mul(input, tensor1), tensor2)


def register_ttl_ternary_function(name, ttl_ternary_function):
    def _torch_ternary(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_torch_function = {
            "mac": torch_mac,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _ternary_validate_input_tensors(operation_name, input, input1, input2, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input1,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
            can_be_a_scalar=True,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input2,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
            can_be_a_scalar=True,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_ternary_validate_input_tensors,
        torch_function=_torch_ternary,
    )
    def ternary_function(
        input: ttnn.Tensor,
        input1: [ttnn.Tensor, float],
        input2: [ttnn.Tensor, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        original_shape = input.shape
        input_tensor = ttnn.unsqueeze_to_4D(input)
        if not _is_scalar(input1) and not _is_scalar(input2):
            input_tensor1 = ttnn.unsqueeze_to_4D(input1)
            input_tensor2 = ttnn.unsqueeze_to_4D(input2)
        elif _is_scalar(input1) and _is_scalar(input2):
            input_tensor1 = input1
            input_tensor2 = input2

        if not isinstance(input_tensor, ttnn.Tensor):
            raise TypeError("Expected input argument to be a ttnn.Tensor")

        if (isinstance(input_tensor1, ttnn.Tensor) and not isinstance(input_tensor2, ttnn.Tensor)) or (
            not isinstance(input_tensor1, ttnn.Tensor) and isinstance(input_tensor2, ttnn.Tensor)
        ):
            raise TypeError("Expected other two inputs as either both tensor or scalar")

        if not ttnn.is_tensor_storage_on_device(input_tensor):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_ternary_function(input_tensor, input_tensor1, input_tensor2)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    ternary_function.__name__ = f"ttnn.{name}"
    ternary_function.decorated_function.__doc__ = f"""{name}(input_tensor: ttnn.Tensor, input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Returns tensor with the {name} of all of elements of the input tensors input, tensor1, tensor2.

        .. math::
            {name.replace('_',' ')}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`
            * :attr:`input_tensor1`
            * :attr:`input_tensor2`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor, tensor1, tensor2)

        {ternary_function.__doc__}

        """
    setattr(THIS_MODULE, name, ternary_function)


TTL_TERNARY_FUNCTIONS = [
    ("mac", ttl.tensor.mac),
]


for ternary_function_name, ttl_ternary_function in TTL_TERNARY_FUNCTIONS:
    register_ttl_ternary_function(ternary_function_name, ttl_ternary_function)


def _is_scalar(value):
    return isinstance(value, (float))


def register_ttl_ternary_function_with_float(name, ttl_ternary_function, op_name, param):
    def _torch_ternary(input_tensor: ttnn.Tensor, parameter, **_):
        import torch

        name_to_torch_function = {
            "addcmul": torch.addcmul,
            "addcdiv": torch.addcdiv,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)

        return torch_function(input_tensor, parameter)

    def _ternary_validate_input_tensors(operation_name, input_tensor, input_tensor1, input_tensor2, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16,),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor1,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16,),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor2,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16,),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_ternary_validate_input_tensors,
        torch_function=_torch_ternary,
    )
    def ternary_function(
        input_tensor: ttnn.Tensor,
        input_tensor1: ttnn.Tensor,
        input_tensor2: ttnn.Tensor,
        parameter: float,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        input_tensor1 = ttnn.unsqueeze_to_4D(input_tensor1)
        input_tensor2 = ttnn.unsqueeze_to_4D(input_tensor2)

        if (
            not isinstance(input_tensor, ttnn.Tensor)
            or not isinstance(input_tensor1, ttnn.Tensor)
            or not isinstance(input_tensor2, ttnn.Tensor)
        ):
            raise TypeError("Expected 3 arguments to be a ttnn.Tensor")

        if not _is_scalar(parameter):
            raise TypeError("Expected one argument to be a float")

        if (
            not ttnn.is_tensor_storage_on_device(input_tensor)
            or not ttnn.is_tensor_storage_on_device(input_tensor1)
            or not ttnn.is_tensor_storage_on_device(input_tensor2)
        ):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_ternary_function(
            input_tensor, input_tensor1, input_tensor2, value=parameter, output_mem_config=memory_config
        )

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    ternary_function.__name__ = f"ttnn.{(name)}"
    ternary_function.decorated_function.__doc__ = f"""{(name)}(input_tensor: ttnn.Tensor, input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor, parameter, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Performs the element-wise {op_name} of tensor1 by tensor2, multiplies the result by the scalar value and adds it to input input.

        .. math::
            {(op_name)}(\\mathrm{{input\\_tensor}}_i  \\; , \\; {param})

        Args:
            * :attr:`input_tensor`
            * :attr:`input_tensor1`
            * :attr:`input_tensor2`
            * :attr:`{param}`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{(name)}(tensor, tensor1, tensor2, {param})

        """
    setattr(THIS_MODULE, name, ternary_function)


TTL_TERNARY_FUNCTIONS_WITH_FLOAT_PARAM = [
    ("addcmul", ttl.tensor.addcmul, "addcmul", "value"),
    ("addcdiv", ttl.tensor.addcdiv, "addcdiv", "value"),
]

for ternary_function_name, ttl_ternary_function, name, param in TTL_TERNARY_FUNCTIONS_WITH_FLOAT_PARAM:
    register_ttl_ternary_function_with_float(ternary_function_name, ttl_ternary_function, name, param)


def register_ttl_ternary_function_where(name, ttl_ternary_function):
    def _torch_ternary(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_torch_function = {
            "where": torch.where,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _ternary_validate_input_tensors(operation_name, input_tensor, input1, input2, *args, **kwargs):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input1,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
            can_be_a_scalar=True,
        )
        ttnn.validate_input_tensor(
            operation_name,
            input2,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
            can_be_a_scalar=True,
        )

    @ttnn.register_operation(
        name=f"ttnn.{name}",
        validate_input_tensors=_ternary_validate_input_tensors,
        torch_function=_torch_ternary,
    )
    def ternary_function(
        predicate: ttnn.Tensor,
        true_value: [ttnn.Tensor, float],
        false_value: [ttnn.Tensor, float],
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        original_shape = predicate.shape
        predicate = ttnn.unsqueeze_to_4D(predicate)
        if not _is_scalar(true_value):
            true_value = ttnn.unsqueeze_to_4D(true_value)
        if not _is_scalar(false_value):
            false_value = ttnn.unsqueeze_to_4D(false_value)

        if not isinstance(predicate, ttnn.Tensor):
            raise TypeError("Expected input_tensor arguments to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(predicate):
            raise RuntimeError("input_tensor must be on device!")

        output_tensor = ttl_ternary_function(predicate, true_value, false_value, output_mem_config=memory_config)

        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    ternary_function.__name__ = f"ttnn.{name}"
    ternary_function.decorated_function.__doc__ = f"""{name}(predicate_tensor: ttnn.Tensor, true_value: [ttnn.Tensor,float], false_value: [ttnn.Tensor,float], *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Perform an ternary {name} operation on two tensors based on predicate input tensor.

        .. math::
            {name.replace('_',' ')}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`predicate_tensor`
            * :attr:`true_value`
            * :attr:`false_value`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor, tensor1, tensor2)

        {ternary_function.__doc__}

        """
    setattr(THIS_MODULE, name, ternary_function)


TTL_TERNARY_FUNCTIONS_WHERE = [
    ("where", ttl.tensor.where),
]


for ternary_function_name, ttl_ternary_function in TTL_TERNARY_FUNCTIONS_WHERE:
    register_ttl_ternary_function_where(ternary_function_name, ttl_ternary_function)


__all__ = []
