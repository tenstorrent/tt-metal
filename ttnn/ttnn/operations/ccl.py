# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

from typing import List

import tt_lib as ttl

import ttnn

import typing

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _validate_input_tensors(operation_name, tensor, *args, **kwargs):
    if tensor.storage_type() not in (ttnn.StorageType.MULTI_DEVICE_HOST, ttnn.StorageType.MULTI_DEVICE):
        raise RuntimeError(f"{operation_name} requires input tensors to be multi-device tensor")


@ttnn.register_operation(
    name="ttnn.all_gather",
    validate_input_tensors=_validate_input_tensors,
    golden_function=None,
)
def all_gather(
    multidevice_tensor: typing.Any,
    dim: int,
    num_links: int,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
):
    device_tensors = [device_tensor for device_tensor in ttnn.get_device_tensors(multidevice_tensor)]
    if len(device_tensors) == 8:
        hamiltonian_ring_indices = [0, 7, 6, 1, 2, 5, 4, 3]
        device_tensors = [device_tensors[i] for i in hamiltonian_ring_indices]
    all_gather_tensor = ttl.tensor.all_gather(device_tensors, dim, num_links, output_mem_config=memory_config)
    return ttnn.aggregate_as_tensor(all_gather_tensor)
