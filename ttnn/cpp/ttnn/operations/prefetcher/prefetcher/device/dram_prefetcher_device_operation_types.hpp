// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::prim {

struct DramPrefetcherParams {
    uint32_t num_layers = 0;
    bool enable_performance_mode = false;
    // global_cb may be either a worker-sender or DRAM-sender GlobalCircularBuffer; the device
    // op picks the right program factory based on global_cb->sender_core_type().
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    // K-block-width in tiles for the DRAM-core path (1 K-tile per push by default; >1 trades
    // fewer-but-bigger pushes for the same total bytes, bounded by DRISC L1).
    uint32_t dram_core_k_block_w_tiles = 1;
};

struct DramPrefetcherInputs {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::prim
