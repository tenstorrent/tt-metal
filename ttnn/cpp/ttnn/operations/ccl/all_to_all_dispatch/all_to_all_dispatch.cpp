// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch.hpp"
#include "device/all_to_all_dispatch_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn::operations::ccl {

std::array<ttnn::Tensor, 2> ExecuteAllToAllDispatch::invoke(
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

    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, axis));
    log_debug(tt::LogOp, "num_links: {}", num_links_);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, axis);
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());
    uint32_t output_concat_dim_ = output_concat_dim.value_or(1);

    const auto [cb_sizes, cb_page_sizes] =
        detail::get_cb_sizes(input_tensor, expert_indices_tensor, expert_mapping_tensor, num_links_, axis);

    AllToAllDispatchDeviceOperation::AllToAllTransferType impl =
        AllToAllDispatchDeviceOperation::AllToAllTransferType::FullPacket;
    uint32_t total_size_bytes = std::accumulate(cb_sizes.begin(), cb_sizes.end(), 0u);
    if (optional_output_tensors.has_value()) {
        const auto& output_tensors = optional_output_tensors.value();
        const auto& output_tensor = output_tensors.at(0);
        const auto& metadata_tensor = output_tensors.at(1);

        if (output_tensor.buffer()->is_l1()) {
            total_size_bytes += output_tensor.buffer()->aligned_size_per_bank();
        }

        if (metadata_tensor.buffer()->is_l1()) {
            total_size_bytes += metadata_tensor.buffer()->aligned_size_per_bank();
        }
    } else if (memory_config_.buffer_type() == tt::tt_metal::BufferType::L1) {
        std::array<ttnn::TensorSpec, 2> specs = AllToAllDispatchDeviceOperation::compute_output_specs(
            AllToAllDispatchDeviceOperation::operation_attributes_t{
                .worker_core_range_set = subdevice_core_range_set,
                .output_mem_config = memory_config_,
                .axis = axis,
                .num_links = num_links_,
                .topology = topology_,
                .impl = impl,
                .output_concat_dim = output_concat_dim_},
            AllToAllDispatchDeviceOperation::tensor_args_t{
                .input_tensor = input_tensor,
                .expert_indices_tensor = expert_indices_tensor,
                .expert_mapping_tensor = expert_mapping_tensor,
                .optional_output_tensors = optional_output_tensors});

        auto alignment = mesh_device->allocator()->get_alignment(memory_config_.buffer_type());
        auto num_banks = mesh_device->allocator()->get_num_banks(memory_config_.buffer_type());
        total_size_bytes +=
            std::accumulate(specs.begin(), specs.end(), 0u, [alignment, num_banks](size_t acc, const auto& spec) {
                return acc + spec.compute_consumed_memory_bytes_per_bank(alignment, num_banks);
            });
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
        num_links_,
        topology_,
        memory_config_,
        subdevice_core_range_set,
        impl,
        output_concat_dim_);
}

}  // namespace ttnn::operations::ccl
