// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_partition.hpp"
#include "device/mesh_partition_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteMeshPartition::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    if (detail::get_cluster_axis_size(input_tensor, cluster_axis) == 1) {
        return input_tensor;
    }

    return ttnn::prim::mesh_partition(
        input_tensor, dim, cluster_axis, memory_config.value_or(input_tensor.memory_config()));
}

}  // namespace ttnn::operations::ccl
