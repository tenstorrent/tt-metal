# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys

import ttnn

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _validate_input_tensors(operation_name, tensor, *args, **kwargs):
    if tensor.storage_type() not in (ttnn.StorageType.MULTI_DEVICE_HOST, ttnn.StorageType.MULTI_DEVICE):
        raise RuntimeError(f"{operation_name} requires input tensors to be multi-device tensor")


all_gather = ttnn.register_operation(validate_input_tensors=_validate_input_tensors)(
    ttnn._ttnn.operations.ccl.all_gather
)
