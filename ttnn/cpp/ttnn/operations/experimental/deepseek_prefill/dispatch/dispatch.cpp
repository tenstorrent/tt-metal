// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.hpp"
#include "device/dispatch_device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

std::array<ttnn::Tensor, 2> dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    const std::optional<ttnn::Tensor>& padding_config,
    const std::optional<ttnn::Tensor>& scales_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    bool use_l1_small_for_semaphores,
    bool fp8_output,
    bool fp8_scaled_input,
    uint32_t num_workers_per_sender) {
    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // A singleton communication axis has no peers, even when other axes have devices.
    // Preserve omitted-axis behavior and the existing local worker pipeline.
    const bool singleton_domain =
        mesh_device->shape().mesh_size() == 1 || (cluster_axis == 0 && mesh_device->shape()[0] == 1);
    const bool local_only = singleton_domain && dispatch_group_size == 1;
    // Validate fabric configuration - only tested values are supported
    TT_FATAL(
        cluster_axis.value_or(0) == 0,
        "cluster_axis must be 0 (current value: {}). Other values are not tested.",
        cluster_axis.value_or(0));
    TT_FATAL(
        (local_only || num_links.value_or(1) >= 1) && num_links.value_or(1) <= 4,
        "num_links must be between 1 and 4, or 0 on a singleton dispatch group (current value: {}).",
        num_links.value_or(1));
    auto topology_ = topology.value_or(tt::tt_fabric::Topology::Linear);
    TT_FATAL(
        topology_ == tt::tt_fabric::Topology::Linear || topology_ == tt::tt_fabric::Topology::Ring,
        "topology must be Linear or Ring. 2D topologies are not supported.");

    std::optional<uint32_t> axis = cluster_axis;
    uint32_t num_links_ =
        local_only ? 0 : (num_links.has_value() ? *num_links : ccl::common::get_num_links(*mesh_device, axis));
    tt::tt_fabric::Topology usable_topology =
        local_only ? tt::tt_fabric::Topology::Linear : ::ttnn::ccl::get_usable_topology(input_tensor, topology_, axis);

    log_debug(tt::LogOp, "num_links={} axis={} topology={}", num_links_, axis, usable_topology);

    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::prefill_dispatch(
        input_tensor,
        indices_tensor,
        expert_offsets_tensor,
        expert_dispatch_table_tensor,
        dispatch_group_size,
        experts_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        metadata_len,
        max_dispatch_buffer_token_size,
        padding_config,
        scales_tensor,
        axis,
        num_links_,
        usable_topology,
        memory_config_,
        subdevice_core_range_set,
        use_l1_small_for_semaphores,
        fp8_output,
        fp8_scaled_input,
        num_workers_per_sender);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
