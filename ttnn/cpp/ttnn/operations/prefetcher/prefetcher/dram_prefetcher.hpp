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

// Worker-sender prefetcher op. Delivers into exactly one of `global_cb` (a worker-sender GCB) or
// `prefetcher_pipes` (worker-sender PrefetcherPipes, one per reader core; the delivery contract is in
// tt_metal/impl/buffers/prefetcher_matmul_design.md, "Delivering into PrefetcherPipes"). The DRAM-core
// counterpart is ttnn.experimental.start_tensor_prefetcher, which delivers into PrefetcherPipes.
ttnn::Tensor dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode = false,
    const ttnn::PrefetcherPipeList& prefetcher_pipes = {});

}  // namespace ttnn
