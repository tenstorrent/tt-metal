// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

// Per-core (row, head) block range of the interleaved nlp_concat_heads program: whole rows per core,
// or (row, head) units when there are fewer rows than cores; contiguous either way.
inline std::pair<uint32_t, uint32_t> nlp_concat_heads_block_range(
    uint32_t core_idx, uint32_t h_tiles, uint32_t batch, uint32_t heads, uint32_t num_cores, bool head_parallel_ok) {
    const uint32_t total_rows = batch * h_tiles;
    if (head_parallel_ok && total_rows < num_cores) {
        const uint32_t total = total_rows * heads;
        return {
            ttnn::runtime_length::balanced_start(core_idx, total, num_cores),
            ttnn::runtime_length::balanced_end(core_idx, total, num_cores)};
    }
    return {
        ttnn::runtime_length::balanced_start(core_idx, total_rows, num_cores) * heads,
        ttnn::runtime_length::balanced_end(core_idx, total_rows, num_cores) * heads};
}
