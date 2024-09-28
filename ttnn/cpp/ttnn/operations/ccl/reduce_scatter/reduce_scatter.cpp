// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_op.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteReduceScatter::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t scatter_dim,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_workers,
    const std::optional<size_t> num_buffers_per_channel) {

    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::ccl::reduce_scatter(input_tensor, scatter_dim, math_op, num_links, out_memory_config, topology, num_workers, num_buffers_per_channel);
}

}  // namespace ttnn::operations::ccl
