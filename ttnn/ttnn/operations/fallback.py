# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import torch

THIS_MODULE = sys.modules[__name__]


def reshape(tensor: ttnn.Tensor, shape) -> ttnn.Tensor:
    torch_tensor = ttnn.to_torch(tensor)
    torch_tensor = torch.reshape(torch_tensor, tuple(shape))
    tensor = ttnn.from_torch(torch_tensor, layout=tensor.layout, dtype=tensor.dtype, device=tensor.device())
    return tensor
