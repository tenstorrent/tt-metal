// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// Worker-sender prefetcher op. Delivers into exactly one of `global_cb` (a worker-sender GCB; a
// DRAM-sender GCB will TT_FATAL with a redirect to ttnn.experimental.start_tensor_prefetcher /
// ttnn.experimental.stop_tensor_prefetcher) or `prefetcher_pipes` (worker-sender PrefetcherPipes, one per
// reader core; see the Python docstring for the delivery contract).
ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode = false,
    const ttnn::PrefetcherPipeList& prefetcher_pipes = {});

}  // namespace ttnn
