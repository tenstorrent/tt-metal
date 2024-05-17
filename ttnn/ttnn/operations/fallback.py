# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import torch
import tt_lib

THIS_MODULE = sys.modules[__name__]


def reshape(tensor: ttnn.Tensor, shape) -> ttnn.Tensor:
    device = tensor.device() if tensor.storage_type() == tt_lib.tensor.StorageType.DEVICE else None
    torch_tensor = ttnn.to_torch(tensor)
    torch_tensor = torch.reshape(torch_tensor, tuple(shape))
    tensor = ttnn.from_torch(torch_tensor, layout=tensor.layout, dtype=tensor.dtype, device=device)
    return tensor
