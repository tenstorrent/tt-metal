# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The generic_op version of toy_scaled_add: the program descriptor built in Python on every call and dispatched
through ttnn.generic_op. It runs the same kernels as the C++ ttnn.toy_scaled_add and checks the same contract,
so the two can be compared call for call."""

from __future__ import annotations

from typing import Optional

import ttnn

from .toy_scaled_add import validate
from .toy_scaled_add_program_descriptor import create_height_sharded_descriptor, create_interleaved_descriptor


def toy_scaled_add_generic(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    alpha: float = 1.0,
    gamma: Optional[ttnn.Tensor] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
    output_tensor: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    dtype, output_memory_config = validate(
        a, b, gamma=gamma, dtype=dtype, memory_config=memory_config, output_tensor=output_tensor
    )
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            a.shape, dtype, ttnn.TILE_LAYOUT, a.device(), output_memory_config
        )

    create = (
        create_height_sharded_descriptor
        if output_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        else create_interleaved_descriptor
    )
    descriptor = create(a, b, gamma, output_tensor, alpha, compute_kernel_config)
    io_tensors = [a, b] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(io_tensors, descriptor)
