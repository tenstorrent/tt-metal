// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_scatter.hpp"
#include "device/split_scatter_op.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteSplitScatter::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<size_t> num_workers,
    const std::optional<size_t> num_buffers_per_channel,
    const ttnn::ccl::Topology topology) {
    return ttnn::operations::experimental::ccl::split_scatter(
        input_tensor, dim, num_links, memory_config, num_workers, num_buffers_per_channel, topology);
}

}  // namespace ttnn::operations::experimental::ccl
