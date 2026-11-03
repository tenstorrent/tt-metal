// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

// Per-core block range (32-index rows) of the interleaved fused-embedding program, derived on device
// from the common row count so the reader, compute and writer agree.
inline std::pair<uint32_t, uint32_t> embeddings_fused_block_range(
    uint32_t core_idx, uint32_t blocks_per_batch, uint32_t batch, uint32_t num_cores) {
    const uint32_t total = batch * blocks_per_batch;
    return {
        ttnn::runtime_length::balanced_start(core_idx, total, num_cores),
        ttnn::runtime_length::balanced_end(core_idx, total, num_cores)};
}
