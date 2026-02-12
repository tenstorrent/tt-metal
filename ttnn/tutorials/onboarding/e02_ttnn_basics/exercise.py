# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Exercise: Implement matmul_add on Tenstorrent device."""

import torch
import ttnn


def matmul_add(device, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute a @ b + c on Tenstorrent device."""
    # TODO: Convert a, b, c to ttnn tensors
    # TODO: Compute matmul and add
    # TODO: Convert result back to torch
    raise NotImplementedError()
