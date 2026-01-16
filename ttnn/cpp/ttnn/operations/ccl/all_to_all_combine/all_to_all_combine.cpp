// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_combine.hpp"
#include "device/all_to_all_combine_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/full/full.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllToAllCombine::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const bool locally_reduced,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<uint32_t>& output_shard_dim,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    uint32_t shard_dim = output_shard_dim.value_or(1);
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, axis));
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, axis);
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());

    // create zeros tensor
    std::optional<ttnn::Tensor> optional_output_tensor_ = optional_output_tensor;
    if (!optional_output_tensor.has_value()) {
        auto output_spec = AllToAllCombineDeviceOperation::compute_output_specs(
            AllToAllCombineDeviceOperation::operation_attributes_t{
                .output_mem_config = memory_config_,
                .axis = axis,
                .num_links = num_links_,
                .topology = topology_,
                .locally_reduced = locally_reduced,
                .output_shard_dim = shard_dim,
            },
            AllToAllCombineDeviceOperation::tensor_args_t{
                .input_tensor = input_tensor,
                .mapping_tensor = expert_mapping_tensor,
                .metadata_tensor = expert_metadata_tensor,
                .optional_output_tensor = optional_output_tensor,
            });
        // currently full only supports tile layout
        ttnn::SmallVector<uint32_t> output_shape;
        output_shape.reserve(output_spec.logical_shape().rank());
        for (size_t i = 0; i < output_spec.logical_shape().rank(); i++) {
            output_shape.push_back(output_spec.logical_shape()[i]);
        }
        auto output_tensor = ttnn::moreh_full(
            output_shape,
            0.0f,
            input_tensor.device(),
            input_tensor.dtype(),
            input_tensor.layout(),
            output_spec.memory_config());
        // set optional_output_tensor to the output tensor
        optional_output_tensor_ = output_tensor;
    }

    return ttnn::prim::all_to_all_combine(
        input_tensor,
        expert_mapping_tensor,
        expert_metadata_tensor,
        num_links_,
        topology_,
        memory_config_,
        axis,
        optional_output_tensor_,
        locally_reduced,
        subdevice_core_range_set,
        shard_dim);
}

}  // namespace ttnn::operations::ccl
