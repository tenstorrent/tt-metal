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
// DRAM-sender GCB will TT_FATAL with a redirect to
// ttnn.experimental.start_tensor_prefetcher / ttnn.experimental.stop_tensor_prefetcher.
ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode = false);

}  // namespace ttnn
