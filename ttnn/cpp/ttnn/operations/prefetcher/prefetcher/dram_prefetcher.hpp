// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <tt-metalium/global_circular_buffer_impl.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include "ttnn/cpp/ttnn/global_circular_buffer.hpp"

namespace ttnn {
namespace operations::dram_prefetcher {

struct ExecuteDramPrefetcher {
    static ttnn::Tensor invoke(
        std::vector<ttnn::Tensor>& tensors,
        const uint32_t num_layers,
        const std::optional<const tt::tt_metal::v1::experimental::GlobalCircularBuffer>& global_cb,
        const std::optional<const ttnn::global_circular_buffer::MultiDeviceGlobalCircularBuffer>& multi_global_cb);
};

}  // namespace operations::dram_prefetcher

constexpr auto dram_prefetcher = ttnn::register_operation_with_auto_launch_op<
    "ttnn::dram_prefetcher",
    ttnn::operations::dram_prefetcher::ExecuteDramPrefetcher>();

}  // namespace ttnn
