// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn {
namespace operations::dram_prefetcher {

struct ExecuteDramPrefetcher {
    static ttnn::Tensor invoke(
        std::vector<ttnn::Tensor>& tensors,
        const Tensor& tensor_addrs,
        const std::optional<const tt::tt_metal::v1::experimental::GlobalCircularBuffer>& global_cb);
};

}  // namespace operations::dram_prefetcher

constexpr auto dram_prefetcher = ttnn::register_operation_with_auto_launch_op<
    "ttnn::dram_prefetcher",
    ttnn::operations::dram_prefetcher::ExecuteDramPrefetcher>();

}  // namespace ttnn
