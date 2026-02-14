# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Solution: matmul_add on Tenstorrent device."""

import torch
import ttnn


def matmul_add(device, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute a @ b + c on Tenstorrent device."""
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_c = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_result = ttnn.matmul(tt_a, tt_b)
    tt_result = ttnn.add(tt_result, tt_c)

    return ttnn.to_torch(tt_result)
