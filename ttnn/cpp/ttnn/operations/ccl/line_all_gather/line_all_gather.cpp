// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/line_all_gather/line_all_gather.hpp"

#include "ttnn/operations/ccl/line_all_gather/device/line_all_gather_op.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteLineAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::OpFabricMode op_fabric_mode) {
    return ttnn::operations::ccl::line_all_gather(input_tensor, dim, num_links, memory_config, op_fabric_mode);
}

ttnn::Tensor ExecuteLineAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
        const ttnn::ccl::OpFabricMode op_fabric_mode) {
    return ttnn::operations::ccl::line_all_gather(
        input_tensor, dim, cluster_axis, mesh_device, num_links, memory_config, op_fabric_mode);
}

}  // namespace ttnn::operations::ccl
