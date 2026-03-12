// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_backward.hpp"
#include "device/all_to_all_dispatch_backward_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/full/full.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllToAllDispatchBackward::invoke(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<uint32_t>& output_shard_dim,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    auto* mesh_device = grad_output.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    uint32_t shard_dim = output_shard_dim.value_or(1);
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, axis));
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(grad_output, topology, axis);
    auto memory_config_ = memory_config.value_or(grad_output.memory_config());

    // Allocate expanded output [dispatch_devices, B_per_device, S, H] filled with zeros.
    // Each dispatch device writes to its own slot (no accumulation conflicts).
    // After the kernel completes, we reduce over dim 0.
    auto expanded_spec = AllToAllDispatchBackwardDeviceOperation::compute_output_specs(
        AllToAllDispatchBackwardDeviceOperation::operation_attributes_t{
            .output_mem_config = memory_config_,
            .axis = axis,
            .num_links = num_links_,
            .topology = topology_,
            .worker_core_range_set = subdevice_core_range_set,
            .output_shard_dim = shard_dim,
        },
        AllToAllDispatchBackwardDeviceOperation::tensor_args_t{
            .grad_output = grad_output,
            .mapping_tensor = expert_mapping_tensor,
            .metadata_tensor = expert_metadata_tensor,
            .optional_output_tensor = optional_output_tensor,
        });

    std::optional<ttnn::Tensor> optional_output_tensor_ = optional_output_tensor;
    if (!optional_output_tensor.has_value()) {
        ttnn::SmallVector<uint32_t> expanded_shape;
        expanded_shape.reserve(expanded_spec.logical_shape().rank());
        for (size_t i = 0; i < expanded_spec.logical_shape().rank(); i++) {
            expanded_shape.push_back(expanded_spec.logical_shape()[i]);
        }
        // The output shape from compute_output_specs is [dispatch_devices, B, S, H]
        auto output_tensor = ttnn::moreh_full(
            expanded_shape,
            0.0f,
            grad_output.device(),
            grad_output.dtype(),
            grad_output.layout(),
            expanded_spec.memory_config());
        optional_output_tensor_ = output_tensor;
    }

    return ttnn::prim::all_to_all_dispatch_backward(
        grad_output,
        expert_mapping_tensor,
        expert_metadata_tensor,
        num_links_,
        topology_,
        memory_config_,
        axis,
        optional_output_tensor_,
        subdevice_core_range_set,
        shard_dim);
}

}  // namespace ttnn::operations::ccl
