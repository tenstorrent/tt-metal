// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/line_all_gather/line_all_gather.hpp"

#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteLineAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    return ttnn::operations::ccl::line_all_gather(input_tensor, dim, num_links, memory_config);
}

ttnn::Tensor ExecuteLineAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const DeviceMesh& device_mesh,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    return ttnn::operations::ccl::line_all_gather(
        input_tensor, dim, cluster_axis, device_mesh, num_links, memory_config);
}

}  // namespace ttnn::operations::ccl
