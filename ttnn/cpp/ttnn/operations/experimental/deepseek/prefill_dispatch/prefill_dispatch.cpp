// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_dispatch.hpp"
#include "device/prefill_dispatch_device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_dispatch {

std::array<ttnn::Tensor, 3> ExecutePrefillDispatch::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weights_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& chip_to_n_routed_expert_offset_tensor,
    uint32_t num_chips,
    uint32_t experts_per_chip,
    uint32_t n_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {


    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // TT_FATAL(sd_id == tt::tt_metal::SubDeviceId{0}, "Currently only subdevice 0 is supported");

    uint32_t axis = 0; // Assuming sharding on the first dimension for now
    // uint32_t num_links_ = common::get_num_links(*mesh_device, axis);
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, axis);
    log_debug(tt::LogOp, "num_links: {}", num_links);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, /*topology*/ std::nullopt, axis);
    log_debug(tt::LogOp, "Selected topology: {}", topology_);
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());


    // const auto [cb_sizes, cb_page_sizes] = detail::get_cb_sizes(input_tensor, weights_tensor, indices_tensor, num_links, axis);
    // TT_THROW("cb_sizes: {}, cb_page_sizes: {}", cb_sizes, cb_page_sizes);
    // TT_THROW("num_links {}; topology {}; memory_config {}; subdevice_id {}", num_links, topology_, memory_config_, sd_id);
    // TODO: Implement parameter validation
    // TODO: Get device and subdevice info
    // TODO: Call device operation

    // Stub implementation - return empty tensors
    // TT_THROW("prefill_dispatch operation not yet implemented");

    return ttnn::prim::prefill_dispatch(
        input_tensor,
        weights_tensor,
        indices_tensor,
        chip_to_n_routed_expert_offset_tensor,
        num_chips,
        experts_per_chip,
        n_routed_experts,
        num_experts_per_tok,
        metadata_len,
        max_dispatched_tokens_per_expert,
        axis,
        num_links,
        topology_,
        memory_config_,
        subdevice_core_range_set);
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch
