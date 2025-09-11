// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void ReduceScatterMinimalAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(
        page_size % input_tensors[0].buffer()->alignment() == 0,
        "reduce_scatter_minimal_async currently requires aligned pages");

    TT_FATAL(this->dim == 3, "reduce_scatter_minimal_async currently only supports reducing on dim 3");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to reduce_scatter_minimal_async need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);

    const auto& input_shape = input_tensor.padded_shape();
    TT_FATAL(
        (input_shape[this->dim] / tt::constants::TILE_WIDTH) % this->ring_size == 0,
        "Error, The number of tiles at input tensor dimension {} should be divisible by ring_size but the number of "
        "tiles is {} and the ring_size is {}",
        this->dim,
        input_shape[this->dim] / tt::constants::TILE_WIDTH,
        this->ring_size);

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    if (output_tensors.size() > 0) {
        TT_FATAL(
            output_tensors.size() == 2,
            "Error, Number of output tensors should be 2 but has {}",
            output_tensors.size());

        // intermediate tensor
        if (output_tensors[0].has_value()) {
            const auto& intermediate_tensor = output_tensors[0].value();

            TT_FATAL(
                intermediate_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                    intermediate_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                    intermediate_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                    intermediate_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Unsupported intermediate tensor memory layout {}.",
                intermediate_tensor.memory_config().memory_layout());

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
            TT_FATAL(
                intermediate_tensor.memory_config() == this->intermediate_mem_config,
                "Error, intermediate tensor memory config should be same as intermediate_mem_config but has {}",
                intermediate_tensor.memory_config());

            // Don't support DRAM block sharding
            if (intermediate_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(
                    intermediate_tensor.memory_config().buffer_type() == BufferType::L1,
                    "We don't support DRAM block sharding");
            }
        }

        // output tensor
        if (output_tensors[1].has_value()) {
            const auto& output_tensor = output_tensors[1].value();

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
                output_tensor.layout() == layout,
                "Error, Output tensor layout should be same as input tensor layout but has {}",
                output_tensor.layout());
            TT_FATAL(
                output_tensor.dtype() == dtype,
                "Error, Output tensor dtype should be same as input tensor dtype but has {}",
                output_tensor.dtype());
            TT_FATAL(
                output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
                "Error, Output tensor page config should be same as input tensor page config but has {}",
                output_tensor.tensor_spec().page_config());
            TT_FATAL(
                output_tensor.memory_config() == this->output_mem_config,
                "Error, Output tensor memory config should be same as output_mem_config but has {}",
                output_tensor.memory_config());

            // check the output tensor size
            auto output_shape = output_tensor.padded_shape();
            auto input_shape = input_tensor.padded_shape();
            TT_FATAL(
                output_shape.size() == input_shape.size(),
                "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
                output_shape.size());
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (i == this->dim) {
                    TT_FATAL(
                        output_shape[i] == input_shape[i] / this->ring_size,
                        "Error, Output tensor shape at dimension {} should be {} but has {}",
                        i,
                        input_shape[i] / this->ring_size,
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

    // Don't support input DRAM block sharding
    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(
            input_tensor.memory_config().buffer_type() == BufferType::L1, "We don't support input DRAM block sharding");
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
    // TODO: FIXME!
    const auto& input_tensor = input_tensors[0];
    auto inter_shape = input_tensor.logical_shape();

    if (this->topology == ccl::Topology::Linear) {
        inter_shape[0] *= 2;
    }

    auto output_shape = input_tensor.logical_shape();
    output_shape[this->dim] /= this->ring_size;
    return {
        TensorSpec(
            inter_shape,
            TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), intermediate_mem_config)),
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
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    const auto& mesh_view = input_tensors[0].device()->get_view();
    if (this->cluster_axis.has_value()) {
        // User specified the cluster-axis. Derive devices based on the current coordinate
        // and the cluster-axis.
        devices_to_use = (this->cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(coord[1])
                                                           : mesh_view.get_devices_on_row(coord[0]);
    } else {
        devices_to_use = devices;
    }
    uint32_t target_ring_size = devices_to_use.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < target_ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(target_ring_size - 1);
            }
            if (i != target_ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    return reduce_scatter_minimal_async(
        input_tensors[0],
        output_tensors[0],
        target_device,
        forward_device,
        backward_device,
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
        this->intermediate_mem_config,
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
    const std::optional<MemoryConfig>& intermeidate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::vector<IDevice*>& devices,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "reduce_scatter_minimal_async op is only supported for Fast Dispatch");

    int32_t rank = input_tensor.logical_shape().rank();
    int32_t scatter_dim = (dim < 0) ? rank + dim : dim;

    // For reduce_scatter_minimal_async_impl, we need to calculate the ring size based on cluster_axis
    // Since we don't have a specific coordinate here, we use the maximum possible devices
    uint32_t num_devices;
    if (cluster_axis.has_value()) {
        auto mesh_device = input_tensor.device();
        TT_FATAL(mesh_device != nullptr, "Mesh device is required when cluster_axis is set");
        const auto& mesh_view = mesh_device->get_view();
        // Use the mesh dimensions to determine the ring size
        num_devices = (cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    } else {
        num_devices = devices.size();
    }

    TT_FATAL(
        num_devices > 1, "reduce_scatter_minimal_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }

    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    // create this semaphore for all cores since we don't know which core will be used for teardown draining
    CoreCoord grid_size = devices[0]->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});

    bool using_persistent_buffers = persistent_output_buffers.has_value();

    std::vector<std::optional<Tensor>> optional_output_tensors =
        using_persistent_buffers
            ? std::vector<std::optional<Tensor>>(persistent_output_buffers->begin(), persistent_output_buffers->end())
            : std::vector<std::optional<Tensor>>{std::nullopt, std::nullopt};

    return tt::tt_metal::operation::run(
               ttnn::ReduceScatterMinimalAsync(
                   devices,
                   scatter_dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   intermeidate_memory_config.value_or(input_tensor.memory_config()),
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
    const std::optional<MemoryConfig>& intermeidate_memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel) {
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    return reduce_scatter_minimal_async_impl(
        input_tensor,
        persistent_output_buffers,
        dim,
        multi_device_global_semaphore,
        barrier_semaphore,
        num_links,
        memory_config,
        intermeidate_memory_config,
        topology,
        sub_device_id,
        devices,
        cluster_axis,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
