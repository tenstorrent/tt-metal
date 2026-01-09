// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_reduce_scatter_minimal_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_minimal_async::detail {

StridedReduceScatterMinimalAsyncDeviceOperation::program_factory_t
StridedReduceScatterMinimalAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Only Ring topology is supported for strided reduce scatter
    TT_FATAL(
        operation_attributes.topology == ttnn::ccl::Topology::Ring,
        "Strided reduce scatter only supports Ring topology, got {}",
        operation_attributes.topology);

    // TODO: Create strided-specific program factory when implementing actual strided logic
    // For now, use the regular Ring reduce scatter factory (wrapped in variant)
    return program_factory_t{reduce_scatter_minimal_async::detail::RingReduceScatterMeshWorkloadFactory{}};
}

void StridedReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Lightweight validation for cache hits
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}

void StridedReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // Topology validation - only Ring supported
    TT_FATAL(
        operation_attributes.topology == ttnn::ccl::Topology::Ring,
        "Strided reduce scatter only supports Ring topology");

    // Common validation (reuse from regular reduce scatter)
    reduce_scatter_minimal_async::detail::reduce_scatter_common_validates(
        input_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        tensor_args.optional_output_tensor);

    const auto layout = input_tensor.layout();
    const auto dtype = input_tensor.dtype();

    // Validate intermediate tensor if provided
    if (tensor_args.optional_intermediate_tensor.has_value()) {
        const auto& intermediate_tensor = tensor_args.optional_intermediate_tensor.value();

        TT_FATAL(intermediate_tensor.storage_type() == StorageType::DEVICE, "Intermediate tensor must be on device");
        TT_FATAL(intermediate_tensor.layout() == layout, "Intermediate tensor layout must match input tensor layout");
        TT_FATAL(intermediate_tensor.dtype() == dtype, "Intermediate tensor dtype must match input tensor dtype");
        TT_FATAL(
            intermediate_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Intermediate tensor page config must match input tensor page config");

        if (operation_attributes.optional_intermediate_mem_config.has_value()) {
            TT_FATAL(
                intermediate_tensor.memory_config() == operation_attributes.optional_intermediate_mem_config.value(),
                "Intermediate tensor memory config must match provided intermediate_mem_config");
        }

        if (intermediate_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                intermediate_tensor.memory_config().buffer_type() == BufferType::L1,
                "DRAM block sharding not supported for intermediate tensor");
        }
    }

    // Validate semaphore count
    constexpr auto num_expected_semaphores = 3;
    TT_FATAL(
        operation_attributes.semaphore.size() == num_expected_semaphores,
        "Expected {} semaphores but got {}",
        num_expected_semaphores,
        operation_attributes.semaphore.size());
}

spec_return_value_t StridedReduceScatterMinimalAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    // For Ring topology, intermediate shape is same as input shape (no 2x like Linear)
    auto inter_shape = input_tensor.logical_shape();
    MemoryConfig intermediate_mem_config =
        operation_attributes.optional_intermediate_mem_config.value_or(input_tensor.memory_config());

    // Output shape has the scatter dimension divided by ring_size
    auto output_shape = input_tensor.logical_shape();
    output_shape[operation_attributes.dim] /= operation_attributes.ring_size;

    return {
        TensorSpec(
            inter_shape,
            TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), intermediate_mem_config)),
        TensorSpec(
            output_shape,
            TensorLayout(
                input_tensor.dtype(),
                input_tensor.tensor_spec().page_config(),
                operation_attributes.output_mem_config)),
    };
}

tensor_return_value_t StridedReduceScatterMinimalAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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

tt::stl::hash::hash_t StridedReduceScatterMinimalAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    return tt::stl::hash::hash_objects(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.optional_intermediate_mem_config,
        operation_attributes.topology,
        operation_attributes.barrier_semaphore.has_value(),
        operation_attributes.using_persistent_buffers,
        operation_attributes.sub_device_id,
        input_tensor.device()->worker_cores(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id),
        operation_attributes.cluster_axis,
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        // Strided-specific attributes
        operation_attributes.tiles_per_chunk,
        operation_attributes.mm_cores_y,
        operation_attributes.mm_block_ht,
        operation_attributes.mm_block_wt,
        // Input tensor properties
        input_tensor.logical_shape(),
        input_tensor.padded_shape(),
        input_tensor.tensor_spec().page_config(),
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.memory_config());
}

}  // namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_minimal_async::detail

namespace ttnn::prim {

ttnn::operations::experimental::ccl::strided_reduce_scatter_minimal_async::detail::
    StridedReduceScatterMinimalAsyncDeviceOperation::tensor_return_value_t
    strided_reduce_scatter_minimal_async(
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
        std::optional<uint32_t> tiles_per_chunk,
        std::optional<uint32_t> mm_cores_y,
        std::optional<uint32_t> mm_block_ht,
        std::optional<uint32_t> mm_block_wt) {
    // TODO: For now, delegate to regular reduce_scatter_minimal_async prim
    // The strided-specific parameters are tracked but not used yet
    log_debug(
        tt::LogOp,
        "strided_reduce_scatter_minimal_async prim: delegating to regular reduce_scatter (strided params tracked "
        "but not used yet)");

    return reduce_scatter_minimal_async(
        input_tensor,
        optional_intermediate_tensor,
        optional_output_tensor,
        dim,
        num_links,
        ring_size,
        std::move(output_mem_config),
        std::move(optional_intermediate_mem_config),
        topology,
        std::move(semaphore),
        std::move(barrier_semaphore),
        using_persistent_buffers,
        sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace ttnn::prim
