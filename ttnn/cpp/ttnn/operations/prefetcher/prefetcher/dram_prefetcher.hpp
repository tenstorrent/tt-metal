// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/types.hpp"

namespace ttnn {

// Worker-sender prefetcher op. `global_cb` must be a worker-sender GCB; passing a
// DRAM-sender GCB will TT_FATAL with a redirect to start_dram_core_prefetcher /
// stop_dram_core_prefetcher (lifecycle API in ttnn/operations/prefetcher).
ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode = false,
    uint32_t dram_core_k_block_w_tiles = 1);

}  // namespace ttnn
