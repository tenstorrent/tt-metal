// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce.hpp"

#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce/device/all_reduce_op.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllReduce::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::operations::reduction::ReduceType math_op,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> num_workers,
    const std::optional<size_t> num_buffers_per_channel) {

    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::experimental::ccl::all_reduce(input_tensor, math_op, num_links, out_memory_config, topology, num_workers, num_buffers_per_channel);
}

}  // namespace ttnn::operations::experimental::ccl
