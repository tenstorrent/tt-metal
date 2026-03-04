// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_combine.hpp"
#include "device/prefill_combine_device_operation.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_combine {

ttnn::Tensor ExecutePrefillCombine::invoke(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& dispatched_metadata,
    const ttnn::Tensor& experts_tok_counter,
    uint32_t num_chips,
    uint32_t experts_per_chip,
    uint32_t num_experts_per_tok,
    uint32_t seq_len_per_chip,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology) {
    // Get device and subdevice info
    auto* mesh_device = dispatched_buffer.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // Validate fabric configuration - only tested values are supported
    TT_FATAL(
        cluster_axis.value_or(0) == 0,
        "cluster_axis must be 0 (current value: {}). Other values are not tested.",
        cluster_axis.value_or(0));
    TT_FATAL(
        num_links.value_or(1) == 1,
        "num_links must be 1 (current value: {}). Other values are not tested.",
        num_links.value_or(1));

    std::optional<uint32_t> axis = cluster_axis;
    uint32_t num_links_ = num_links.value_or(ccl::common::get_num_links(*mesh_device, axis));
    auto topology_ = topology.value_or(tt::tt_fabric::Topology::Linear);
    tt::tt_fabric::Topology usable_topology = ::ttnn::ccl::get_usable_topology(dispatched_buffer, topology_, axis);

    log_debug(tt::LogOp, "num_links={} axis={} topology={}", num_links_, axis, usable_topology);

    auto memory_config_ = memory_config.value_or(dispatched_buffer.memory_config());

    // Call the primitive operation
    return ttnn::prim::prefill_combine(
        dispatched_buffer,
        dispatched_metadata,
        experts_tok_counter,
        num_chips,
        experts_per_chip,
        num_experts_per_tok,
        seq_len_per_chip,
        axis,
        num_links_,
        usable_topology,
        memory_config_,
        subdevice_core_range_set);
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine
