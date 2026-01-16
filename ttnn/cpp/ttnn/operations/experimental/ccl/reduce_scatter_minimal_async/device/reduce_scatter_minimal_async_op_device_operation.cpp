// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail {

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
    reduce_scatter_common_validates(
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

spec_return_value_t ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs(
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

tensor_return_value_t ReduceScatterMinimalAsyncDeviceOperation::create_output_tensors(
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

tt::stl::hash::hash_t ReduceScatterMinimalAsyncDeviceOperation::compute_program_hash(
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
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

// Common validation function implementation
void reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    const auto page_size = input_tensor.buffer()->page_size();
    TT_FATAL(
        topology == ttnn::ccl::Topology::Ring || topology == ttnn::ccl::Topology::Linear,
        "Topology must be either Ring or Linear");
    TT_FATAL(
        page_size % input_tensor.buffer()->alignment() == 0,
        "reduce_scatter_minimal_async currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffers on device");
    TT_FATAL(num_links > 0, "num_links must be greater than 0");

    const auto& rank = input_tensor.logical_shape().rank();
    TT_FATAL(rank > 1, "reduce_scatter currently supports rank 2 tensors at minimum");
    TT_FATAL(dim < rank, "Invalid scatter dim {} for rank {} tensor", dim, rank);

    const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, rank));
    const auto& input_shape = input_tensor.padded_shape();

    if (normalized_dim == 2 || normalized_dim == 3) {
        uint32_t tile_size = normalized_dim == 2 ? tt::constants::TILE_HEIGHT : tt::constants::TILE_WIDTH;
        TT_FATAL(
            (input_shape[dim] / tile_size) % ring_size == 0,
            "Number of tiles at dimension {} must be divisible by ring_size",
            dim);
    } else {
        TT_FATAL(input_shape[dim] % ring_size == 0, "Dimension {} must be divisible by ring_size", dim);
    }

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout");

    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            input_tensor.memory_config().buffer_type() == BufferType::L1,
            "DRAM block sharding not supported for input");
    }

    if (optional_output_tensor.has_value()) {
        const auto& output_tensor = optional_output_tensor.value();

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout");

        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor must be on device");
        TT_FATAL(
            output_tensor.layout() == input_tensor.layout(), "Output tensor layout must match input tensor layout");
        TT_FATAL(output_tensor.dtype() == input_tensor.dtype(), "Output tensor dtype must match input tensor dtype");
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Output tensor page config must match input tensor page config");
        TT_FATAL(
            output_tensor.memory_config() == memory_config,
            "Output tensor memory config must match provided memory_config");

        // Check output tensor size
        auto output_shape = output_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Output tensor must have same number of dimensions as input tensor");

        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == dim) {
                TT_FATAL(
                    output_shape[i] == input_shape[i] / ring_size,
                    "Output tensor dimension {} must be input dimension / ring_size",
                    i);
            } else {
                TT_FATAL(output_shape[i] == input_shape[i], "Output tensor dimension {} must match input dimension", i);
            }
        }

        if (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                output_tensor.memory_config().buffer_type() == BufferType::L1,
                "DRAM block sharding not supported for output");
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail

namespace ttnn::prim {

ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail::ReduceScatterMinimalAsyncDeviceOperation::
    tensor_return_value_t
    reduce_scatter_minimal_async(
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
        std::optional<uint32_t> num_buffers_per_channel) {
    using OperationType = ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail::
        ReduceScatterMinimalAsyncDeviceOperation;
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
        num_buffers_per_channel};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, optional_intermediate_tensor, optional_output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
