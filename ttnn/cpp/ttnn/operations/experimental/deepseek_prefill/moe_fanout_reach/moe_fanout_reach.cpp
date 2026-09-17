// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_fanout_reach.hpp"

#include <tt-metalium/sub_device.hpp>

#include "device/moe_fanout_reach_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach {

ttnn::Tensor moe_fanout_reach(
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_dispatch_table,
    const ttnn::Tensor& global_dispatch_offsets,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t dispatch_group_size,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
    // Every core this op may occupy. The model runs the dispatch stage on a sub-device that is one row
    // of the compute grid, and this op runs beside it on the same routing metadata, so the caller says
    // which cores are free exactly as it does for `dispatch_fabric2d`. Defaulting to the first
    // sub-device means no sub-device manager loaded gives the whole grid, which is what a test gets.
    auto* mesh_device = indices_tensor.device();
    const auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const tt::tt_metal::CoreRangeSet universe =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    return ttnn::prim::moe_fanout_reach(
        mesh_device,
        indices_tensor,
        expert_dispatch_table,
        global_dispatch_offsets,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_group_size,
        max_dispatch_buffer_token_size,
        cluster_axis,
        memory_config.value_or(
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}),
        universe);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach
