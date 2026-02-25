// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher.hpp"
#include <optional>
#include "device/dram_prefetcher_device_operation.hpp"

namespace ttnn::operations::dram_prefetcher {

Tensor ExecuteDramPrefetcher::invoke(
    std::vector<ttnn::Tensor>& tensors,
    const uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const bool enable_performance_mode) {
    return ttnn::prim::dram_prefetcher(tensors, num_layers, global_cb, enable_performance_mode);
}

}  // namespace ttnn::operations::dram_prefetcher
