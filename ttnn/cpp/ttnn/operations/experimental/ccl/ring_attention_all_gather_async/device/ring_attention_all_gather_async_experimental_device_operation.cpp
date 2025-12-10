// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_experimental_device_operation.hpp"
#include "ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::ccl {

RingAttentionAllGatherAsyncExperimentalDeviceOperation::program_factory_t
RingAttentionAllGatherAsyncExperimentalDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RingAttentionAllGatherAsyncDeviceOperation::select_program_factory(operation_attributes, tensor_args);
}

void RingAttentionAllGatherAsyncExperimentalDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_hit(operation_attributes, tensor_args);
}

void RingAttentionAllGatherAsyncExperimentalDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_miss(operation_attributes, tensor_args);
}

RingAttentionAllGatherAsyncExperimentalDeviceOperation::spec_return_value_t
RingAttentionAllGatherAsyncExperimentalDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RingAttentionAllGatherAsyncDeviceOperation::compute_output_specs(operation_attributes, tensor_args);
}

RingAttentionAllGatherAsyncExperimentalDeviceOperation::tensor_return_value_t
RingAttentionAllGatherAsyncExperimentalDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RingAttentionAllGatherAsyncDeviceOperation::create_output_tensors(operation_attributes, tensor_args);
}

tt::stl::hash::hash_t RingAttentionAllGatherAsyncExperimentalDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash(operation_attributes, tensor_args);
}

std::tuple<
    RingAttentionAllGatherAsyncExperimentalDeviceOperation::operation_attributes_t,
    RingAttentionAllGatherAsyncExperimentalDeviceOperation::tensor_args_t>
RingAttentionAllGatherAsyncExperimentalDeviceOperation::invoke(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API without 2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensors[0].logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<std::optional<Tensor>> optional_output_tensors;
    optional_output_tensors.reserve(persistent_output_buffer.size());
    for (size_t i = 0; i < persistent_output_buffer.size(); ++i) {
        optional_output_tensors.push_back(persistent_output_buffer[i]);
    }

    return RingAttentionAllGatherAsyncDeviceOperation::invoke(
        input_tensors,
        persistent_output_buffer,
        gather_dim,
        multi_device_global_semaphore,
        num_links,
        num_devices,  // This is the ring size
        cluster_axis,
        mesh_device,
        memory_config,
        topology,
        sub_device_id);
}

}  // namespace ttnn::operations::experimental::ccl
