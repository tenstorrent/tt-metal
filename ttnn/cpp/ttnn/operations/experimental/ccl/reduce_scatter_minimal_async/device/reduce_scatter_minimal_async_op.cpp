// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

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

void ReduceScatterMinimalAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    std::optional<Tensor> optional_output_tensor =
        (output_tensors.size() == 2 && output_tensors[1].has_value()) ? output_tensors[1] : std::nullopt;
    reduce_scatter_common_validates(
        input_tensor,
        this->topology,
        this->dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        optional_output_tensor);
    const auto layout = input_tensor.layout();
    const auto dtype = input_tensor.dtype();
    if (!output_tensors.empty()) {
        TT_FATAL(
            output_tensors.size() == 2,
            "Error, Number of output tensors should be 2 but has {}",
            output_tensors.size());

        // intermediate tensor
        if (output_tensors[0].has_value()) {
            const auto& intermediate_tensor = output_tensors[0].value();

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

            if (this->optional_intermediate_mem_config.has_value()) {
                TT_FATAL(
                    intermediate_tensor.memory_config() == this->optional_intermediate_mem_config.value(),
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
    }

    // Each direction has a ready semaphore and there's a global sync semaphore, per link.
    const auto num_expected_semaphores = 3;
    TT_FATAL(
        semaphore.size() == num_expected_semaphores,
        "Error, semaphore size should be {} but has {}",
        num_expected_semaphores,
        semaphore.size());
}

std::vector<ttnn::TensorSpec> ReduceScatterMinimalAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto inter_shape = input_tensor.logical_shape();

    MemoryConfig adjusted_intermediate_mem_config,
        intermediate_mem_config = optional_intermediate_mem_config.value_or(input_tensor.memory_config());
    if (this->topology == ccl::Topology::Linear) {
        inter_shape[0] *= 2;

        // need to adjust memory config taken from input tensor
        if (intermediate_mem_config.is_sharded() && !optional_intermediate_mem_config.has_value()) {
            auto intermediate_shard_spec = intermediate_mem_config.shard_spec().value();
            intermediate_shard_spec.shape[0] *= 2;
            adjusted_intermediate_mem_config = intermediate_mem_config.with_shard_spec(intermediate_shard_spec);
        }
    } else {
        adjusted_intermediate_mem_config = intermediate_mem_config;
    }

    auto output_shape = input_tensor.logical_shape();
    output_shape[this->dim] /= this->ring_size;
    return {
        TensorSpec(
            inter_shape,
            TensorLayout(
                input_tensor.dtype(), input_tensor.tensor_spec().page_config(), adjusted_intermediate_mem_config)),
        TensorSpec(
            output_shape,
            TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config)),
    };
}

std::vector<Tensor> ReduceScatterMinimalAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    auto tensor_specs = compute_output_specs(input_tensors);

    ttnn::Tensor intermediate_buffer = optional_output_tensors.at(0).has_value()
                                           ? optional_output_tensors.at(0).value()
                                           : create_device_tensor(tensor_specs[0], input_tensors.at(0).device());

    ttnn::Tensor output_buffer = optional_output_tensors.at(1).has_value()
                                     ? optional_output_tensors.at(1).value()
                                     : create_device_tensor(tensor_specs[1], input_tensors.at(0).device());

    return {intermediate_buffer, output_buffer};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks ReduceScatterMinimalAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks ReduceScatterMinimalAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at {} is called", coord);
    uint32_t target_ring_size = ::ttnn::ccl::get_topological_dimension(input_tensors[0], this->cluster_axis);

    log_debug(tt::LogOp, "Getting forward neighbor for {}", coord);
    const std::optional<MeshCoordinate> forward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, 1, this->topology, this->cluster_axis);

    log_debug(tt::LogOp, "Getting backward neighbor for {}", coord);
    const std::optional<MeshCoordinate> backward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, -1, this->topology, this->cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    log_debug(tt::LogOp, "Getting device index for {}", coord);
    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(input_tensors[0], coord, this->cluster_axis);
    log_debug(tt::LogOp, "Device index for {} is {}", coord, device_index);

    return reduce_scatter_minimal_async(
        input_tensors[0],
        output_tensors[0],
        coord,
        forward_coord,
        backward_coord,
        output_tensors[1],
        this->dim,
        this->num_links,
        target_ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->barrier_semaphore,
        this->using_persistent_buffers,
        this->sub_device_id,
        this->chunks_per_sync,
        this->num_workers_per_link,
        this->num_buffers_per_channel);
}

tt::tt_metal::operation::Hash ReduceScatterMinimalAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    return tt::tt_metal::operation::hash_operation<ReduceScatterMinimalAsync>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->optional_intermediate_mem_config,
        this->topology,
        this->barrier_semaphore.has_value(),
        this->using_persistent_buffers,
        this->sub_device_id.has_value(),
        this->sub_device_id.has_value()
            ? input_tensors[0].device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, this->sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        this->cluster_axis,
        this->chunks_per_sync,
        this->num_workers_per_link,
        this->num_buffers_per_channel,
        input_tensors);
}

namespace operations {
namespace experimental {
namespace ccl {

namespace {
Tensor reduce_scatter_minimal_async_impl(
    const Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& optional_intermediate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel) {
    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    // For reduce_scatter_minimal_async_impl, we need to calculate the ring size based on cluster_axis
    // Since we don't have a specific coordinate here, we use the maximum possible devices
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices > 1, "reduce_scatter_minimal_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    // create this semaphore for all cores since we don't know which core will be used for teardown draining
    CoreCoord grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    bool using_persistent_buffers = persistent_output_buffers.has_value();

    std::vector<std::optional<Tensor>> optional_output_tensors =
        using_persistent_buffers
            ? std::vector<std::optional<Tensor>>(persistent_output_buffers->begin(), persistent_output_buffers->end())
            : std::vector<std::optional<Tensor>>{std::nullopt, std::nullopt};

    return tt::tt_metal::operation::run(
               ttnn::ReduceScatterMinimalAsync(
                   scatter_dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   optional_intermediate_memory_config,
                   ccl_topology,
                   multi_device_global_semaphore,
                   barrier_semaphore,
                   using_persistent_buffers,
                   sub_device_id,
                   cluster_axis,
                   chunks_per_sync,
                   num_workers_per_link,
                   num_buffers_per_channel),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(1);
}
}  // namespace

Tensor reduce_scatter_minimal_async(
    const Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& intermediate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    return reduce_scatter_minimal_async_impl(
        input_tensor,
        persistent_output_buffers,
        dim,
        multi_device_global_semaphore,
        barrier_semaphore,
        num_links,
        memory_config,
        intermediate_memory_config,
        topology,
        sub_device_id,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
