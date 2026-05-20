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

// `global_cb` may be a worker-sender or DRAM-sender GlobalCircularBuffer; the op picks
// the right program factory based on the experimental
// `tt::tt_metal::experimental::sender_core_type(*global_cb)` query.
ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode = false,
    uint32_t dram_core_k_block_w_tiles = 1);

}  // namespace ttnn
