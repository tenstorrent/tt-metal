# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.block_gather / ttnn.bringup.block_keys: the two data-movement steps of an EXACT block pre-selected
top-k (deepseek_v41_flash_d_p PB.BM, INDEXER_BLOCK_SELECT): per row, the top-K blocks of B adjacent scores by block max
(a top-k over the block maxima), then the top-k keys over only those blocks' scores.

    block_gather   scores [.., R, W] bf16 RM, ids [.., R, K] uint32 RM (block ids, e.g. a top-k's output; >= W / B,
                   0xFFFFFFFF included: none) -> [.., R, K B] bf16 RM: row r = the K blocks' B scores each, in ids'
                   order (a missing block: B x -inf)
    block_keys     pos [.., R, KP] uint32 RM (positions in block_gather's row, e.g. a top-k of it; 0xFFFFFFFF: none),
                   ids [.., R, K] uint32 RM (the same ids) -> [.., R, KP] uint32 RM: ids[pos / B] B + pos % B (none:
                   0xFFFFFFFF)

Why not ttnn.gather: its ROW_MAJOR kernel holds each core's whole input row in L1 (a 1M-key row is 2 MiB) and reads it
once a core; here a block is one 64-byte DRAM read. One data-movement kernel each (NoC 0), rows split over the cores by
common runtime args (no per-core args, DS41-145). block_gather, block_keys and block_max_bf16 are native C++ operations
(their programs cached on shapes only, the addresses rewritten on a hit).
"""

from __future__ import annotations

import ttnn

GATHER_PARTS = 4  # block_gather: a row's blocks split into this many work units (both RISCs of every core)
SENTINEL = 0xFFFFFFFF


# block_gather(scores, ids, *, block, output_tensor=None, memory_config=None, max_grid=None): the native C++
# operation (ttnn/cpp/ttnn/operations/block_gather), which plans, validates and caches its program itself.
block_gather = ttnn.block_gather


# block_keys(pos, ids, *, block, width, output_tensor=None, memory_config=None, max_grid=None): the native C++
# operation (ttnn/cpp/ttnn/operations/block_keys), which validates, plans and caches its program itself.
block_keys = ttnn.block_keys


# block_max_bf16(block_max, *, factor=1, n=None, n_tensor=None, output_tensor=None, memory_config=None,
# max_grid=None, strip_major=False): the native C++ operation (ttnn/cpp/ttnn/operations/block_max_bf16).
#   strip_major (PB.FBM): block_max [.., S NB / 8, 256] bf16 ROW_MAJOR, strip-major in 512-byte pages of 8 blocks ->
#   output_tensor [.., 32 S, NB] bf16 ROW_MAJOR (required), the first n blocks (rounded up to 8); factor 1.
#   Else: block_max [.., R, NI] fp32 ROW_MAJOR (bf16 values widened) -> [.., R, NI / factor] bf16 ROW_MAJOR, column b =
#   the max of input columns [b factor, (b + 1) factor), for the first n output columns (rounded up to 8). n a runtime
#   value or n_tensor ([1] uint32 ROW_MAJOR DRAM, read on the device). factor a power of two <= 16.
block_max_bf16 = ttnn.block_max_bf16
