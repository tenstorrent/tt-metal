// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <tt-metalium/global_circular_buffer.hpp>
#include "device/dram_prefetcher_op.hpp"

namespace ttnn {
namespace operations::dram_prefetcher {

struct ExecuteDramPrefetcher {
    static ttnn::Tensor invoke(
        std::vector<ttnn::Tensor>& tensors,
        const uint32_t num_layers,
        const std::optional<const GlobalCircularBuffer>& global_cb,
        const bool enable_performance_mode = false);
};

}  // namespace operations::dram_prefetcher

constexpr auto dram_prefetcher =
    ttnn::register_operation<"ttnn::dram_prefetcher", ttnn::operations::dram_prefetcher::ExecuteDramPrefetcher>();

}  // namespace ttnn
