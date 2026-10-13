// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async_generic/device/all_to_all_async_generic_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor all_to_all_async_generic(
    const ttnn::Tensor& input_tensor,
    int32_t in_dim,
    int32_t out_dim,
    const std::optional<Tensor>& persistent_output_buffer,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    return ttnn::prim::all_to_all_async_generic(
        input_tensor,
        persistent_output_buffer,
        in_dim,
        out_dim,
        num_links,
        memory_config,
        topology.value_or(tt::tt_fabric::get_fabric_topology()),
        subdevice_id,
        cluster_axis);
}

}  // namespace ttnn::experimental
