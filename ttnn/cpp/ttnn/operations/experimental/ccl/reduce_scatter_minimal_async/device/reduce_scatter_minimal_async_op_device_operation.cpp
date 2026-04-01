// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ReduceScatterMinimalAsyncDeviceOperation::program_factory_t
ReduceScatterMinimalAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
        return RingReduceScatterMeshWorkloadFactory{};
    }
    TT_FATAL(operation_attributes.topology == ttnn::ccl::Topology::Linear, "Topology must be Ring or Linear");
    return LineReduceScatterMeshWorkloadFactory{};
}

void ReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Lightweight validation for cache hits
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}

void ReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Common validation
    ttnn::experimental::ccl::reduce_scatter_common_validates(
        input_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        tensor_args.optional_output_tensor);

    // Validate intermediate tensor if provided
    if (tensor_args.optional_intermediate_tensor.has_value()) {
        ttnn::experimental::ccl::validate_intermediate_tensor(
            input_tensor,
            tensor_args.optional_intermediate_tensor.value(),
            operation_attributes.optional_intermediate_mem_config);
    }

    // Validate semaphore count
    constexpr auto num_expected_semaphores = 3;
    TT_FATAL(
        operation_attributes.semaphore.size() == num_expected_semaphores,
        "Expected {} semaphores but got {}",
        num_expected_semaphores,
        operation_attributes.semaphore.size());
}

std::vector<ttnn::TensorSpec> ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs(
    const ReduceScatterMinimalAsyncParams& operation_attributes, const ReduceScatterMinimalAsyncInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto inter_shape = input_tensor.padded_shape();

    MemoryConfig adjusted_intermediate_mem_config,
        intermediate_mem_config =
            operation_attributes.optional_intermediate_mem_config.value_or(input_tensor.memory_config());

    if (operation_attributes.topology == ttnn::ccl::Topology::Linear) {
        inter_shape[0] *= 2;

        // Adjust memory config for sharded tensors
        if (intermediate_mem_config.is_sharded() &&
            !operation_attributes.optional_intermediate_mem_config.has_value()) {
            auto intermediate_shard_spec = intermediate_mem_config.shard_spec().value();
            intermediate_shard_spec.shape[0] *= 2;
            adjusted_intermediate_mem_config = intermediate_mem_config.with_shard_spec(intermediate_shard_spec);
        } else {
            adjusted_intermediate_mem_config = intermediate_mem_config;
        }
    } else {
        adjusted_intermediate_mem_config = intermediate_mem_config;
    }

    auto output_shape = input_tensor.logical_shape();
    output_shape[operation_attributes.dim] /= operation_attributes.ring_size;

    return {
        TensorSpec(
            inter_shape,
            TensorLayout(
                input_tensor.dtype(), input_tensor.tensor_spec().page_config(), adjusted_intermediate_mem_config)),
        TensorSpec(
            output_shape,
            TensorLayout(
                input_tensor.dtype(),
                input_tensor.tensor_spec().page_config(),
                operation_attributes.output_mem_config)),
    };
}

std::vector<Tensor> ReduceScatterMinimalAsyncDeviceOperation::create_output_tensors(
    const ReduceScatterMinimalAsyncParams& operation_attributes, const ReduceScatterMinimalAsyncInputs& tensor_args) {
    auto tensor_specs = compute_output_specs(operation_attributes, tensor_args);
    const auto& input_tensor = tensor_args.input_tensor;

    ttnn::Tensor intermediate_buffer = tensor_args.optional_intermediate_tensor.has_value()
                                           ? tensor_args.optional_intermediate_tensor.value()
                                           : create_device_tensor(tensor_specs[0], input_tensor.device());

    ttnn::Tensor output_buffer = tensor_args.optional_output_tensor.has_value()
                                     ? tensor_args.optional_output_tensor.value()
                                     : create_device_tensor(tensor_specs[1], input_tensor.device());

    return {intermediate_buffer, output_buffer};
}

ttsl::hash::hash_t ReduceScatterMinimalAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "ReduceScatterMinimalAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<ReduceScatterMinimalAsyncDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.optional_intermediate_mem_config,
        operation_attributes.topology,
        operation_attributes.barrier_semaphore.has_value(),
        operation_attributes.using_persistent_buffers,
        operation_attributes.cluster_axis,
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        operation_attributes.compute_kernel_config,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> reduce_scatter_minimal_async(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    MemoryConfig output_mem_config,
    std::optional<MemoryConfig> optional_intermediate_mem_config,
    ttnn::ccl::Topology topology,
    std::vector<GlobalSemaphore> semaphore,
    std::optional<GlobalSemaphore> barrier_semaphore,
    bool using_persistent_buffers,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::ReduceScatterMinimalAsyncDeviceOperation;
    const auto resolved_sub_device_id = sub_device_id.value_or(input_tensor.device()->get_sub_device_ids().at(0));

    auto operation_attributes = OperationType::operation_attributes_t{
        dim,
        num_links,
        ring_size,
        std::move(output_mem_config),
        std::move(optional_intermediate_mem_config),
        topology,
        std::move(semaphore),
        std::move(barrier_semaphore),
        using_persistent_buffers,
        resolved_sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        compute_kernel_config};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, optional_intermediate_tensor, optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
