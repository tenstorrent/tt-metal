// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher.hpp"
#include <optional>

#include "device/dram_prefetcher_op.hpp"
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::operations::dram_prefetcher {

Tensor ExecuteDramPrefetcher::invoke(
    std::vector<ttnn::Tensor>& tensors,
    const uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const bool enable_performance_mode) {
    return tt::tt_metal::operation::run(DramPrefetcher{global_cb, num_layers, enable_performance_mode}, tensors).at(0);
}

}  // namespace ttnn::operations::dram_prefetcher
