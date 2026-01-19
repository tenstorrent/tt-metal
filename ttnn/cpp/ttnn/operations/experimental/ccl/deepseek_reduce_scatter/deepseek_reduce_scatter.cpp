// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include "deepseek_reduce_scatter.hpp"
#include "device/deepseek_reduce_scatter_device_operation.hpp"

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteDeepseekReduceScatter::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    const ttnn::MemoryConfig& output_memory_config,
    int32_t dim,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis) {
    uint32_t scatter_dim = (dim < 0) ? dim + input_tensors.at(0).logical_shape().rank() : (uint32_t)dim;

    // Call the prim operation
    std::vector<ttnn::Tensor> result =
        ttnn::prim::deepseek_reduce_scatter(input_tensors, output_memory_config, scatter_dim, num_links, cluster_axis);

    // Return the output tensor (first 8 are intermediates)
    return result.at(8);
}

}  // namespace ttnn::operations::experimental::ccl
