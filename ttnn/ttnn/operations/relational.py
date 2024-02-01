# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import tt_lib as ttl

import ttnn.core as ttnn


THIS_MODULE = sys.modules[__name__]

__all__ = []


def register_ttl_relational_function(name, ttl_relational_function, op_function):
    def _torch_relational(input_tensor: ttnn.Tensor, **_):
        import torch

        name_to_torch_function = {
            "gtz": torch.gt,
            "ltz": torch.lt,
            "gez": torch.ge,
            "lez": torch.le,
            "nez": torch.ne,
            "eqz": torch.eq,
        }
        torch_function = name_to_torch_function[name]
        input_tensor = ttnn.to_torch(input_tensor)
        return torch_function(input_tensor)

    def _relational_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
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
        validate_input_tensors=_relational_validate_input_tensors,
        torch_function=_torch_relational,
    )
    def relational_function(
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

        ttl_output_tensor = ttl_relational_function(ttl_input_tensor, output_mem_config=memory_config)

        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn.reshape(output_tensor, original_shape)
        return output_tensor

    relational_function.__name__ = f"ttnn.{name}"
    relational_function.__doc__ = f"""{name}(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

        Returns tensor with the {op_function} of all of the elements of the input tensor :attr:`input_tensor` element-wise.

        .. math::
            {name}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor`

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.{name}(tensor)

        """
    setattr(THIS_MODULE, name, relational_function)


TTL_RELATIONAL_FUNCTIONS = [
    ("gtz", ttl.tensor.gtz, "greater than zero"),
    ("ltz", ttl.tensor.ltz, "less than zero"),
    ("gez", ttl.tensor.gez, "greater than or equal to zero"),
    ("lez", ttl.tensor.lez, "less than or equal to zero"),
    ("nez", ttl.tensor.nez, "not equal to zero"),
    ("eqz", ttl.tensor.eqz, "equal to zero"),
]


for relational_function_name, ttl_relational_function, name in TTL_RELATIONAL_FUNCTIONS:
    register_ttl_relational_function(relational_function_name, ttl_relational_function, name)

__all__ = []
