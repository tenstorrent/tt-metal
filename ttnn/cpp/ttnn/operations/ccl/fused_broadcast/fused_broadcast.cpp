// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_broadcast.hpp"
#include "ttnn/operations/ccl/fused_broadcast/device/fused_broadcast_op.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteFusedBroadcast::invoke(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& mesh_shape,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t num_links) {
    auto mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required for fused_broadcast operation");

    // Default memory config
    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());

    // Calculate ring size and index
    uint32_t ring_size = mesh_shape[0] * mesh_shape[1];

    return ttnn::operations::ccl::fused_broadcast_impl(
        input_tensor, root_coord, mesh_shape, num_links, ring_size, topology, output_mem_config, sub_device_id);
}  // namespace operations::ccl

}  // namespace ttnn::operations::ccl
