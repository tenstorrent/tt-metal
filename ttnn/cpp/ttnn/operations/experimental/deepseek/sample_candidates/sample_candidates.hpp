// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// The top-k candidates of one row of logits as ttnn.sampling takes them, in one program.
//
// logits: [1, 1, 1, V] BFLOAT16 TILE, DRAM interleaved (row 0 is read).
// inv_temp: [1, 1, 1, 1] BFLOAT16 TILE (element 0 scales the values).
// index_offset: [1, 1, 1, 1] INT32 ROW_MAJOR, added to every index (a TP rank's first vocab column).
//
// Returns {values [1, 1, 1, 2k] BFLOAT16 TILE: the k largest logits, descending (ties by lower index), times
// inv_temp, then k fill_value; indices [1, 1, 1, 2k] INT32 ROW_MAJOR: their columns + index_offset, then 0},
// both DRAM interleaved. num_workers cores each keep the top k of their share of the tiles; core 0 merges them.
std::vector<Tensor> decode_sample_candidates(
    const Tensor& logits,
    const Tensor& inv_temp,
    const Tensor& index_offset,
    uint32_t k = 32,
    float fill_value = -1.0e9F,
    uint32_t num_workers = 64);

}  // namespace ttnn::experimental::deepseek
