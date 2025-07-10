// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "all_to_all_dispatch.hpp"
#include "device/all_to_all_dispatch_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::ccl {

std::array<ttnn::Tensor, 2> ExecuteAllToAllDispatch::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<GlobalSemaphore>& global_semaphore) {
    auto mesh_device = input_tensor.mesh_device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    TT_FATAL(
        global_semaphore.has_value(),
        "Global semaphore is required for all_to_all_dispatch due to limitations in trace");
    const auto [cb_sizes, cb_page_sizes] =
        detail::get_cb_sizes(input_tensor, expert_indices_tensor, expert_mapping_tensor, axis);

    AllToAllDispatchDeviceOperation::AllToAllTransferType impl =
        AllToAllDispatchDeviceOperation::AllToAllTransferType::FullPacket;
    uint32_t total_size_bytes = std::accumulate(cb_sizes.begin(), cb_sizes.end(), 0u);
    if (optional_output_tensors.has_value()) {
        auto output_tensors = optional_output_tensors.value();
        auto output_tensor = output_tensors.at(0);
        auto metadata_tensor = output_tensors.at(1);

        if (output_tensor.buffer()->is_l1()) {
            total_size_bytes += output_tensor.buffer()->aligned_size_per_bank();
        }

        if (metadata_tensor.buffer()->is_l1()) {
            total_size_bytes += metadata_tensor.buffer()->aligned_size_per_bank();
        }
    }
    uint32_t available_l1_space =
        mesh_device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).largest_free_block_bytes;
    if (available_l1_space < total_size_bytes) {
        impl = AllToAllDispatchDeviceOperation::AllToAllTransferType::PageByPage;
    }

    log_debug(tt::LogOp, "remaining L1 space: {}", available_l1_space - total_size_bytes);
    log_debug(
        tt::LogOp,
        "impl: {}",
        impl == AllToAllDispatchDeviceOperation::AllToAllTransferType::PageByPage ? "PageByPage" : "FullPacket");

    return ttnn::prim::all_to_all_dispatch(
        input_tensor,
        expert_indices_tensor,
        expert_mapping_tensor,
        axis,
        optional_output_tensors,
        num_links,
        topology,
        memory_config.value_or(input_tensor.memory_config()),
        sd_id,
        global_semaphore,
        impl);
}

}  // namespace ttnn::operations::ccl
