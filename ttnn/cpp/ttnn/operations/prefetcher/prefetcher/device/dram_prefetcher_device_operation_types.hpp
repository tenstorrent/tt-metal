// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/dram_sender_global_circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::prim {

struct DramPrefetcherParams {
    uint32_t num_layers = 0;
    bool enable_performance_mode = false;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    // DRAM-core mode (Blackhole): runs the prefetcher as a DRISC kernel on
    // programmable DRAM cores, fed via GDDR DMA. Mutually exclusive with `global_cb`.
    bool run_on_dram_cores = false;
    std::optional<const tt::tt_metal::experimental::DramSenderGlobalCircularBuffer> dram_sender_global_cb;
    // K-block-width in tiles for the DRAM-core path (1 K-tile per push by default; >1 trades
    // fewer-but-bigger pushes for the same total bytes, bounded by DRISC L1).
    uint32_t dram_core_k_block_w_tiles = 1;
};

struct DramPrefetcherInputs {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::prim
