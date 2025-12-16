// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_selective_tilize.hpp"
#include "device/all_to_all_dispatch_selective_tilize_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn::operations::experimental::ccl {

std::array<ttnn::Tensor, 2> ExecuteAllToAllDispatchSelectiveTilize::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<uint32_t>& output_concat_dim) {
    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    uint32_t num_links_ = num_links.value_or(::ttnn::operations::ccl::common::get_num_links(*mesh_device, axis));
    log_debug(tt::LogOp, "num_links: {}", num_links_);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, axis);
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());
    uint32_t output_concat_dim_ = output_concat_dim.value_or(1);

    // Only FullPacket mode is supported for this operation
    AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllTransferType impl =
        AllToAllDispatchSelectiveTilizeDeviceOperation::AllToAllTransferType::FullPacket;

    return ttnn::prim::all_to_all_dispatch_selective_tilize(
        input_tensor,
        expert_indices_tensor,
        expert_mapping_tensor,
        axis,
        optional_output_tensors,
        num_links_,
        topology_,
        memory_config_,
        subdevice_core_range_set,
        impl,
        output_concat_dim_);
}

}  // namespace ttnn::operations::experimental::ccl
