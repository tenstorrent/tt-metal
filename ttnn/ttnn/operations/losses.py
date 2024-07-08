# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn
import ttnn._ttnn.deprecated as ttl

__all__ = []


def register_ttl_loss_function(name, ttl_loss_function):
    def _golden_function(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, loss_mode: str, **_):
        import torch

        name_to_golden_function_function = {
            "mse_loss": torch.nn.MSELoss,
            "l1_loss": torch.nn.L1Loss,
        }
        torch_function = name_to_golden_function_function[name]
        return torch_function(reduction=loss_mode)(input_tensor_a, input_tensor_b)

    doc = f"""{name}(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, loss_mode: str, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

            Applies {name} to :attr:`input_tensor_a` and :attr:`input_tensor_b` with loss_mode :attr:`loss_mode`.

            .. math::
                {name.replace('_',' ')}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor_a`
                * :attr:`input_tensor_b`
                * :attr:`loss_mode`

            Example::

                >>> tensor1 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> tensor2 = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = ttnn.{name}(tensor1, tensor2, mode)

            """

    @ttnn.register_python_operation(name=f"ttnn.{name}", golden_function=_golden_function, doc=doc)
    def loss_function(
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        loss_mode: str,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        input_tensor_a = ttnn.unsqueeze_to_4D(input_tensor_a)
        input_tensor_b = ttnn.unsqueeze_to_4D(input_tensor_b)

        if not isinstance(input_tensor_a, ttnn.Tensor) or not isinstance(input_tensor_b, ttnn.Tensor):
            raise TypeError("Expected both arguments to be a ttnn.Tensor")

        if not ttnn.is_tensor_storage_on_device(input_tensor_a) or not ttnn.is_tensor_storage_on_device(input_tensor_b):
            raise RuntimeError("Both input_tensor must be on device!")

        if loss_mode == "none":
            mode = ttl.tensor.LossReductionMode.NONE
        if loss_mode == "sum":
            mode = ttl.tensor.LossReductionMode.SUM
        if loss_mode == "mean":
            mode = ttl.tensor.LossReductionMode.MEAN

        output_tensor = ttl_loss_function(input_tensor_a, input_tensor_b, mode, output_mem_config=memory_config)

        output_tensor = ttnn.unsqueeze_to_4D(output_tensor)
        return output_tensor


TTL_UNARY_FUNCTIONS = [
    ("mse_loss", ttl.tensor.mseloss),
    ("l1_loss", ttl.tensor.maeloss),
]


for unary_function_name, ttl_unary_function in TTL_UNARY_FUNCTIONS:
    register_ttl_loss_function(unary_function_name, ttl_unary_function)


__all__ = []
