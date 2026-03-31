// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_reduce_scatter_async_op_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail {

StridedReduceScatterAsyncDeviceOperation::program_factory_t
StridedReduceScatterAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(
        operation_attributes.topology == ttnn::ccl::Topology::Ring,
        "strided_reduce_scatter_async only supports Ring topology");
    return RingStridedReduceScatterMeshWorkloadFactory{};
}

void StridedReduceScatterAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Lightweight validation for cache hits
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}

void StridedReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss(
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
        const auto& intermediate_tensor = tensor_args.optional_intermediate_tensor.value();

        ttnn::experimental::ccl::validate_intermediate_tensor(
            input_tensor, intermediate_tensor, operation_attributes.optional_intermediate_mem_config);

        // Ring kernels address with no batch offset, so the intermediate must be single-batch.
        auto expected_inter_shape = input_tensor.logical_shape();
        expected_inter_shape[0] = 1;
        TT_FATAL(
            intermediate_tensor.logical_shape() == expected_inter_shape,
            "Intermediate tensor shape must be single-batch (batch dim 1), expected {} but got {}",
            expected_inter_shape,
            intermediate_tensor.logical_shape());
    }

    // The reader/writer kernels assert dim==3 (W); reject any other dim at host to produce a clear error.
    TT_FATAL(
        operation_attributes.dim == 3,
        "strided_reduce_scatter_async only supports dim=3 (W), but got dim={}",
        operation_attributes.dim);

    // The ring scatter kernel does not loop over the C dimension, so C must be 1
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4 && input_tensor.logical_shape()[1] == 1,
        "strided_reduce_scatter_async requires a 4D tensor with C=1, but got shape {}",
        input_tensor.logical_shape());

    // mm_block_ht and mm_block_wt are used as divisors in the program factory
    TT_FATAL(operation_attributes.mm_block_ht > 0, "mm_block_ht must be > 0, got {}", operation_attributes.mm_block_ht);
    TT_FATAL(operation_attributes.mm_block_wt > 0, "mm_block_wt must be > 0, got {}", operation_attributes.mm_block_wt);

    // Validate semaphore count
    constexpr auto num_expected_semaphores = 3;
    TT_FATAL(
        operation_attributes.semaphore.size() == num_expected_semaphores,
        "Expected {} semaphores but got {}",
        num_expected_semaphores,
        operation_attributes.semaphore.size());
}

spec_return_value_t StridedReduceScatterAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto inter_shape = input_tensor.logical_shape();

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
        // Ring: intermediate buffer holds one batch at a time (kernels use no batch offset).
        inter_shape[0] = 1;
        if (intermediate_mem_config.is_sharded() &&
            !operation_attributes.optional_intermediate_mem_config.has_value()) {
            auto intermediate_shard_spec = intermediate_mem_config.shard_spec().value();
            intermediate_shard_spec.shape[0] = 1;
            adjusted_intermediate_mem_config = intermediate_mem_config.with_shard_spec(intermediate_shard_spec);
        } else {
            adjusted_intermediate_mem_config = intermediate_mem_config;
        }
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

tensor_return_value_t StridedReduceScatterAsyncDeviceOperation::create_output_tensors(
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

tt::stl::hash::hash_t StridedReduceScatterAsyncDeviceOperation::compute_program_hash(
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
        operation_attributes.sub_device_id.has_value(),
        operation_attributes.sub_device_id.has_value()
            ? input_tensor.device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        operation_attributes.cluster_axis,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        operation_attributes.mm_cores_y,
        operation_attributes.mm_block_ht,
        operation_attributes.mm_block_wt,
        operation_attributes.mm_N_full_block_wt,
        operation_attributes.chunk_width_in_mm_blocks,
        input_tensor.logical_shape(),
        input_tensor.padded_shape(),
        input_tensor.tensor_spec().page_config(),
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.memory_config());
}

}  // namespace ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail

namespace ttnn::prim {

ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterAsyncDeviceOperation::
    tensor_return_value_t
    strided_reduce_scatter_async(
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
        std::optional<uint32_t> num_workers_per_link,
        std::optional<uint32_t> num_buffers_per_channel,
        std::optional<uint32_t> mm_cores_y,
        uint32_t mm_block_ht,
        uint32_t mm_block_wt,
        std::optional<uint32_t> mm_N_full_block_wt,
        std::optional<uint32_t> chunk_width_in_mm_blocks) {
    using OperationType = ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::
        StridedReduceScatterAsyncDeviceOperation;
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
        num_workers_per_link,
        num_buffers_per_channel,
        mm_cores_y,
        mm_block_ht,
        mm_block_wt,
        mm_N_full_block_wt,
        chunk_width_in_mm_blocks};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, optional_intermediate_tensor, optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
