# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.pair_pool: the softmax-gated pooling of row pairs in one program (deepseek_v41_flash_d_p P.24,
TINY_KERNELS; the compressor's ratio-2 pooling, reference compressor kv * softmax(gate) summed over each pair).

    x [.., R, 2 hd] fp32 TILE = [kv | gate] per row  ->  out [.., R / 2, hd] fp32 TILE
    out[t] = kv[2t] w0 + kv[2t + 1] w1,  (w0, w1) = softmax(gate[2t], gate[2t + 1]) per column

The operation is the native C++ ttnn.pair_pool (ttnn/cpp/ttnn/operations/pair_pool): it checks the contract (refusing
with AssertionError), plans the units over the grid (common runtime args only) and caches the program; a cache hit
rewrites only the input and output addresses.
"""

from __future__ import annotations

import ttnn

pair_pool = ttnn.pair_pool
