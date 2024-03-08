# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from ttnn.types import (
    DataType,
    Layout,
    DEVICE_STORAGE_TYPE,
    MULTI_DEVICE_STORAGE_TYPE,
    Tensor,
)


def validate_input_tensor(
    operation_name,
    tensor: Tensor,
    *,
    ranks: Tuple[int, ...],
    dtypes: Tuple[DataType, ...],
    layouts: Tuple[Layout, ...],
    can_be_on_device: bool,
    can_be_on_cpu: bool,
    can_be_a_scalar: bool = False,
    is_optional: bool = False,
):
    if is_optional and tensor is None:
        return

    ranks = set(ranks)
    dtypes = set(dtypes)
    layouts = set(layouts)

    if can_be_a_scalar:
        if isinstance(tensor, (int, float)):
            return
        elif not isinstance(tensor, Tensor):
            raise RuntimeError(
                f"{operation_name}: Tensor must be of type int, float or ttnn.Tensor, but got {type(tensor)}"
            )
    else:
        if not isinstance(tensor, Tensor):
            raise RuntimeError(f"{operation_name}: Tensor must be of type ttnn.Tensor, but got {type(tensor)}")

    if len(tensor.shape) not in ranks:
        raise RuntimeError(f"{operation_name}: Tensor must be of rank {ranks}, but got {len(tensor.shape)}")

    if tensor.dtype not in dtypes:
        raise RuntimeError(f"{operation_name}: Tensor must be of type {dtypes}, but got {tensor.dtype}")

    if tensor.layout not in layouts:
        raise RuntimeError(f"{operation_name}: Tensor must be of layout {layouts}, but got {tensor.layout}")

    if can_be_on_device and can_be_on_cpu:
        pass
    elif can_be_on_device:
        if tensor.storage_type() not in (DEVICE_STORAGE_TYPE, MULTI_DEVICE_STORAGE_TYPE):
            raise RuntimeError(f"{operation_name}: Tensor must be on device!")
    elif can_be_on_cpu:
        if tensor.storage_type() in (DEVICE_STORAGE_TYPE, MULTI_DEVICE_STORAGE_TYPE):
            raise RuntimeError(f"{operation_name}: Tensor must be on host!")
    else:
        raise RuntimeError(f"{operation_name}: Tensor must be on host or device!")

    if not tensor.is_allocated():
        raise RuntimeError(f"{operation_name}: Tensor must be allocated!")


__all__ = []
