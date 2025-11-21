// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include <tt-metalium/core_coord.hpp>

using namespace ttnn;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async {

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
        topology == ::ttnn::ccl::Topology::Ring || topology == ::ttnn::ccl::Topology::Linear,
        "topology must be Ring or Linear");
    TT_FATAL(
        page_size % input_tensor.buffer()->alignment() == 0,
        "reduce_scatter_minimal_async currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to reduce_scatter_minimal_async need to be allocated in buffers on device!");
    TT_FATAL(num_links > 0, "Error, num_links should be more than 0 but has {}", num_links);

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "input_tensor must have a buffer");

    const auto& rank = input_tensor.logical_shape().rank();

    TT_FATAL(rank > 1, "reduce_scatter currently supports rank 2 tensors at minimum");
    TT_FATAL(dim < rank, "Invalid scatter dim {} for rank {} tensor", dim, rank);

    const uint32_t normalized_dim = std::get<0>(composite_common::normalize_dim_4d(dim, rank));
    const auto& input_shape = input_tensor.padded_shape();
    if (normalized_dim == 2 || normalized_dim == 3) {
        uint32_t tile_size = normalized_dim == 2 ? tt::constants::TILE_HEIGHT : tt::constants::TILE_WIDTH;
        TT_FATAL(
            (input_shape[dim] / tile_size) % ring_size == 0,
            "Error, The number of tiles at input tensor dimension {} should be divisible by ring_size but the number "
            "of tiles is {} and the ring_size is {}",
            dim,
            input_shape[dim] / tile_size,
            ring_size);
    } else {
        TT_FATAL(
            input_shape[dim] % ring_size == 0,
            "Error, input tensor dimension {} should be divisible by ring_size but is {} and the ring_size is {}",
            dim,
            input_shape[dim],
            ring_size);
    }

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            input_tensor.memory_config().buffer_type() == BufferType::L1, "We don't support input DRAM block sharding");
    }

    if (optional_output_tensor.has_value()) {
        const auto& output_tensor = optional_output_tensor.value();

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout {}.",
            output_tensor.memory_config().memory_layout());

        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Operands to reduce_scatter_minimal_async need to be on device!");
        TT_FATAL(
            output_tensor.layout() == input_tensor.layout(),
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.layout());
        TT_FATAL(
            output_tensor.dtype() == input_tensor.dtype(),
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.tensor_spec().page_config());
        TT_FATAL(
            output_tensor.memory_config() == memory_config,
            "Error, Output tensor memory config {} should be same as output_mem_config {}",
            output_tensor.memory_config(),
            memory_config);

        // check the output tensor size
        auto output_shape = output_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == dim) {
                TT_FATAL(
                    output_shape[i] == input_shape[i] / ring_size,
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] / ring_size,
                    output_shape[i]);
            } else {
                TT_FATAL(
                    output_shape[i] == input_shape[i],
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i],
                    output_shape[i]);
            }
        }

        // Don't support DRAM block sharding
        if (output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                output_tensor.memory_config().buffer_type() == BufferType::L1,
                "We don't support output DRAM block sharding");
        }
    }
}

ReduceScatterMinimalAsyncDeviceOperation::program_factory_t
ReduceScatterMinimalAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    if (operation_attributes.topology == ::ttnn::ccl::Topology::Ring) {
        return program::ring::RingReduceScatterMinimalAsyncProgramFactory{};
    } else {
        return program::line::LineReduceScatterMinimalAsyncProgramFactory{};
    }
}

void ReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void ReduceScatterMinimalAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    std::optional<Tensor> optional_output_tensor;
    if (tensor_args.persistent_output_buffers.has_value() && tensor_args.persistent_output_buffers->size() >= 2) {
        optional_output_tensor = tensor_args.persistent_output_buffers->at(1);
    }

    reduce_scatter_common_validates(
        input_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        optional_output_tensor);

    const auto layout = input_tensor.layout();
    const auto dtype = input_tensor.dtype();

    if (tensor_args.persistent_output_buffers.has_value()) {
        TT_FATAL(
            tensor_args.persistent_output_buffers->size() == 2,
            "Error, Number of output tensors should be 2 but has {}",
            tensor_args.persistent_output_buffers->size());

        // intermediate tensor
        const auto& intermediate_tensor = tensor_args.persistent_output_buffers->at(0);

        TT_FATAL(
            intermediate_tensor.storage_type() == StorageType::DEVICE,
            "Operands to reduce_scatter_minimal_async need to be on device!");
        TT_FATAL(
            intermediate_tensor.layout() == layout,
            "Error, intermediate tensor layout should be same as input tensor layout but has {}",
            intermediate_tensor.layout());
        TT_FATAL(
            intermediate_tensor.dtype() == dtype,
            "Error, intermediate tensor dtype should be same as input tensor dtype but has {}",
            intermediate_tensor.dtype());
        TT_FATAL(
            intermediate_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, intermediate tensor page config should be same as input tensor page config but has {}",
            intermediate_tensor.tensor_spec().page_config());

        if (operation_attributes.optional_intermediate_mem_config.has_value()) {
            TT_FATAL(
                intermediate_tensor.memory_config() == operation_attributes.optional_intermediate_mem_config.value(),
                "Error, intermediate tensor memory config should be same as intermediate_mem_config but has {}",
                intermediate_tensor.memory_config());
        }

        // Don't support DRAM block sharding
        if (intermediate_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                intermediate_tensor.memory_config().buffer_type() == BufferType::L1,
                "We don't support DRAM block sharding");
        }
    }

    // Each direction has a ready semaphore and there's a global sync semaphore, per link.
    const auto num_expected_semaphores = 3;
    TT_FATAL(
        operation_attributes.semaphore.size() == num_expected_semaphores,
        "Error, semaphore size should be {} but has {}",
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
    if (operation_attributes.topology == ::ttnn::ccl::Topology::Linear) {
        inter_shape[0] *= 2;

        // need to adjust memory config taken from input tensor
        if (intermediate_mem_config.is_sharded() &&
            !operation_attributes.optional_intermediate_mem_config.has_value()) {
            auto intermediate_shard_spec = intermediate_mem_config.shard_spec().value();
            intermediate_shard_spec.shape[0] *= 2;
            adjusted_intermediate_mem_config = intermediate_mem_config.with_shard_spec(intermediate_shard_spec);
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

    ttnn::Tensor intermediate_buffer =
        (tensor_args.persistent_output_buffers.has_value() && tensor_args.persistent_output_buffers->size() >= 1)
            ? tensor_args.persistent_output_buffers->at(0)
            : create_device_tensor(tensor_specs[0], tensor_args.input_tensor.device());

    ttnn::Tensor output_buffer =
        (tensor_args.persistent_output_buffers.has_value() && tensor_args.persistent_output_buffers->size() >= 2)
            ? tensor_args.persistent_output_buffers->at(1)
            : create_device_tensor(tensor_specs[1], tensor_args.input_tensor.device());

    return {intermediate_buffer, output_buffer};
}

tt::stl::hash::hash_t ReduceScatterMinimalAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<ReduceScatterMinimalAsyncDeviceOperation>(
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
            ? tensor_args.input_tensor.device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        operation_attributes.cluster_axis,
        operation_attributes.chunks_per_sync,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        tensor_args.input_tensor);
}

std::tuple<
    ReduceScatterMinimalAsyncDeviceOperation::operation_attributes_t,
    ReduceScatterMinimalAsyncDeviceOperation::tensor_args_t>
ReduceScatterMinimalAsyncDeviceOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<std::vector<Tensor>>& persistent_output_buffers,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& intermediate_memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices > 1, "reduce_scatter_minimal_async op will only work for num_devices > 1, but has {}", num_devices);

    bool using_persistent_buffers = persistent_output_buffers.has_value();

    operation_attributes_t operation_attributes{
        scatter_dim,
        num_links,
        num_devices,
        memory_config.value_or(input_tensor.memory_config()),
        intermediate_memory_config,
        topology,
        multi_device_global_semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel};

    tensor_args_t tensor_args{input_tensor, persistent_output_buffers};

    return {operation_attributes, tensor_args};
}

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async
