// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_op.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn {

void AllGatherAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Error, Input tensor size should be 1 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(
        (tt::tt_metal::hal::get_arch_name() != "blackhole") ||
            (input_tensor.memory_config().buffer_type() != BufferType::DRAM) ||
            !this->use_all_gather_async_llama_sharded,
        "This kernel does not support blackhole dram as it does not use an accessor to get the noc address as needed "
        "by the fabric api");
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported input tensor memory layout {}.",
        input_tensor.memory_config().memory_layout());

    AllGatherAsyncVersion version = select_version(input_tensors[0]);

    if (!output_tensors.empty() and output_tensors[0].has_value()) {
        TT_FATAL(
            output_tensors.size() <= 1,
            "Error, Number of output tensors should be at most 1 but has {}",
            output_tensors.size());
        const auto& output_tensor = output_tensors[0];

        TT_FATAL(
            output_tensor.value().storage_type() == StorageType::DEVICE,
            "Operands to all_gather need to be on device!");
        TT_FATAL(
            output_tensor.value().layout() == layout,
            "Error, Output tensor layout should be same as input tensor layout but has {}",
            output_tensor.value().layout());
        TT_FATAL(
            output_tensor.value().dtype() == dtype,
            "Error, Output tensor dtype should be same as input tensor dtype but has {}",
            output_tensor.value().dtype());
        TT_FATAL(
            output_tensor.value().tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Error, Output tensor page config should be same as input tensor page config but has {}",
            output_tensor.value().tensor_spec().page_config());
        TT_FATAL(
            output_tensor.value().memory_config() == this->output_mem_config,
            "Error, Output tensor memory config should be same as output_mem_config but has {}",
            output_tensor.value().memory_config());

        TT_FATAL(
            output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Unsupported output tensor memory layout {}.",
            output_tensor.value().memory_config().memory_layout());

        // check the output tensor size
        auto output_shape = output_tensor.value().padded_shape();
        auto input_shape = input_tensor.padded_shape();
        TT_FATAL(
            output_shape.size() == input_shape.size(),
            "Error, Output tensor shape should have same number of dimensions as input tensor but has {}",
            output_shape.size());
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i == this->dim) {
                TT_FATAL(
                    output_shape[i] <= input_shape[i] * this->ring_size,
                    "Error, Output tensor shape at dimension {} should be {} but has {}",
                    i,
                    input_shape[i] * this->ring_size,
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

        if (version == AllGatherAsyncVersion::MINIMAL_DEFAULT) {
            // Checks specific to the MINIMAL_DEFAULT case

            // Don't support output DRAM block sharding
            if (output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(
                    output_tensor.value().memory_config().buffer_type() == BufferType::L1,
                    "We don't support output DRAM block sharding");
            }
        } else {
            // Checks specific to cases that are not MINIMAL_DEFAULT

            TT_FATAL(
                output_tensor.value().memory_config().memory_layout() == input_tensor.memory_config().memory_layout(),
                "Error, Output tensor memory layout should be same as input tensor memory layout but has {}",
                output_tensor.value().memory_config().memory_layout());
        }
    }

    // Checks specific to the MINIMAL_DEFAULT case
    if (version == AllGatherAsyncVersion::MINIMAL_DEFAULT) {
        // Don't support input DRAM block sharding
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                input_tensor.memory_config().buffer_type() == BufferType::L1,
                "We don't support input DRAM block sharding");
        }
        TT_FATAL(input_tensor.logical_shape().rank() >= 2, "AllGatherAsync requires tensor of rank 2 or greater");
    } else {
        TT_FATAL(input_tensor.logical_shape().rank() == 4, "Llama specific all_gather requires tensor of rank 4");
    }
}

std::vector<ttnn::TensorSpec> AllGatherAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[this->dim] *= this->ring_size;
    return {TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config))};
}

std::vector<Tensor> AllGatherAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

AllGatherAsyncVersion AllGatherAsync::select_version(const Tensor& input_tensor) const {
    // Check for minimal sharded case
    if (this->use_all_gather_async_llama_sharded) {
        TT_FATAL(
            !this->reverse_order,
            "Reversed all-gather (reverse_order=true) is not yet supported with llama-optimized variants "
            "(use_all_gather_async_llama_sharded=true). Please use the regular all_gather_async API instead of "
            "all_gather_async_reversed.");
        return AllGatherAsyncVersion::LLAMA_MINIMAL_SHARDED;
    } else {
        TT_FATAL(this->semaphore.size() == 2, "Default implementation requires 2 semaphores");
        return AllGatherAsyncVersion::MINIMAL_DEFAULT;
    }
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGatherAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    AllGatherAsyncVersion version = select_version(input_tensors[0]);

    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(input_tensors[0], coord, this->cluster_axis);

    std::optional<MeshCoordinate> forward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, 1, this->topology, this->cluster_axis);

    std::optional<MeshCoordinate> backward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, -1, this->topology, this->cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");
    log_trace(tt::LogOp, "version: {}", static_cast<uint32_t>(version));

    switch (version) {
        case AllGatherAsyncVersion::LLAMA_MINIMAL_SHARDED:
            log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_async_llama_sharded is called");
            return all_gather_async_llama_sharded(
                input_tensors[0],
                coord,
                forward_coord,
                backward_coord,
                output_tensors[0],
                this->dim,
                this->num_links,
                this->ring_size,
                device_index,
                this->topology,
                this->semaphore.at(0),
                this->barrier_semaphore,
                this->using_persistent_buffers,
                this->sub_device_id,
                this->use_optimal_ccl_for_llama);

        case AllGatherAsyncVersion::MINIMAL_DEFAULT:
        default:
            log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_async_minimal_default is called");
            return all_gather_async_minimal_default(
                input_tensors[0],
                coord,
                forward_coord,
                backward_coord,
                output_tensors[0],
                this->dim,
                this->num_links,
                this->ring_size,
                device_index,
                this->topology,
                this->semaphore,
                this->barrier_semaphore,
                this->using_persistent_buffers,
                this->sub_device_id,
                this->chunks_per_sync,
                this->num_workers_per_link,
                this->num_buffers_per_channel,
                this->reverse_order,
                this->sub_core_grid);
    }
}

tt::tt_metal::operation::Hash AllGatherAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "AllGatherAsync::compute_program_hash is called");

    const ttnn::Tensor& input_tensor = input_tensors[0];

    return tt::tt_metal::operation::hash_operation<AllGatherAsync>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        this->sub_device_id.has_value(),
        this->sub_device_id.has_value()
            ? input_tensors[0].device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, this->sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        this->barrier_semaphore.has_value(),
        this->using_persistent_buffers,
        this->chunks_per_sync,
        this->num_workers_per_link,
        this->num_buffers_per_channel,
        this->use_all_gather_async_llama_sharded,
        this->use_optimal_ccl_for_llama,
        this->reverse_order,
        input_tensor);
}

namespace operations::experimental::ccl {

namespace {
Tensor all_gather_async_impl(
    const Tensor& input_tensor,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool use_all_gather_async_llama_sharded,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required");
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, std::nullopt);

    TT_FATAL(num_devices > 1, "all_gather_async op will only work for num_devices > 1, but has {}", num_devices);

    bool using_persistent_buffers = false;

    return tt::tt_metal::operation::run(
               ttnn::AllGatherAsync(
                   dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   topology,
                   multi_device_global_semaphore,
                   sub_device_id,
                   /*cluster_axis=*/std::nullopt,
                   use_all_gather_async_llama_sharded,
                   use_optimal_ccl_for_llama,
                   barrier_semaphore,
                   using_persistent_buffers,
                   std::nullopt,
                   std::nullopt,
                   std::nullopt,
                   reverse_order,
                   sub_core_grid),
               {input_tensor})
        .at(0);
}

Tensor all_gather_async_impl(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<uint32_t>& cluster_axis,
    bool use_all_gather_async_llama_sharded,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<uint32_t>& chunks_per_sync,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    TT_FATAL(input_tensor.device() != nullptr, "Mesh device is required");

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    TT_FATAL(num_devices > 1, "all_gather_async op will only work for num_devices > 1, but has {}", num_devices);

    log_debug(tt::LogOp, "DEBUG: creating line_fabric with num devices: {}, num links: {}", num_devices, num_links);
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    bool using_persistent_buffers = persistent_output_buffer.has_value();

    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_buffer};

    return tt::tt_metal::operation::run(
               ttnn::AllGatherAsync(
                   dim,
                   num_links,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   topology,
                   multi_device_global_semaphore,
                   sub_device_id,
                   cluster_axis,
                   use_all_gather_async_llama_sharded,
                   use_optimal_ccl_for_llama,
                   barrier_semaphore,
                   using_persistent_buffers,
                   chunks_per_sync,
                   num_workers_per_link,
                   num_buffers_per_channel,
                   reverse_order,
                   sub_core_grid),
               {input_tensor},
               {},
               optional_output_tensors)
        .at(0);
}

Tensor all_gather_async_impl(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool use_all_gather_async_llama_sharded,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    bool using_persistent_buffers = persistent_output_tensor.has_value();

    std::vector<std::optional<Tensor>> optional_output_tensors = {persistent_output_tensor};

    return tt::tt_metal::operation::run(
               ttnn::AllGatherAsync{
                   gather_dim,
                   num_preferred_links.has_value() ? num_preferred_links.value() : 1,
                   num_devices,
                   memory_config.value_or(input_tensor.memory_config()),
                   topology,
                   multi_device_global_semaphore,
                   sub_device_id,
                   cluster_axis,
                   use_all_gather_async_llama_sharded,
                   use_optimal_ccl_for_llama,
                   barrier_semaphore,
                   using_persistent_buffers,
                   std::nullopt,
                   std::nullopt,
                   std::nullopt,
                   reverse_order,
                   sub_core_grid},
               {input_tensor},
               {},
               optional_output_tensors)
        .at(0);
}
}  // namespace

Tensor all_gather_async(
    const Tensor& input_tensor,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool use_optimal_ccl_for_llama,
    bool use_all_gather_async_llama_sharded,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return all_gather_async_impl(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        sub_device_id,
        use_all_gather_async_llama_sharded,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        reverse_order,
        sub_core_grid);
}

Tensor all_gather_async(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    bool use_optimal_ccl_for_llama,
    bool use_all_gather_async_llama_sharded,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return all_gather_async_impl(
        input_tensor,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        sub_device_id,
        cluster_axis,
        use_all_gather_async_llama_sharded,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        reverse_order,
        sub_core_grid);
}

std::vector<Tensor> all_gather_async(
    const std::vector<Tensor>& input_tensors,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> cluster_axis,
    bool use_optimal_ccl_for_llama,
    bool use_all_gather_async_llama_sharded,
    const std::optional<std::vector<GlobalSemaphore>>& barrier_semaphore,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        output_tensors.push_back(all_gather_async_impl(
            input_tensors[i],
            persistent_output_buffer,
            dim,
            // 0 = forward link, 1 = backward link, i = device index
            {multi_device_global_semaphore.at(0).global_semaphores.at(i),
             multi_device_global_semaphore.at(1).global_semaphores.at(i)},
            num_links,
            memory_config,
            topology,
            sub_device_id,
            cluster_axis,
            use_all_gather_async_llama_sharded,
            use_optimal_ccl_for_llama,
            barrier_semaphore.has_value() ? std::optional<GlobalSemaphore>(barrier_semaphore.value()[i]) : std::nullopt,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            reverse_order,
            sub_core_grid));
    }
    return output_tensors;
}

Tensor all_gather_async(
    const Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    bool use_all_gather_async_llama_sharded,
    bool use_optimal_ccl_for_llama,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool reverse_order,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return all_gather_async_impl(
        input_tensor,
        dim,
        cluster_axis,
        mesh_device,
        topology,
        multi_device_global_semaphore,
        persistent_output_tensor,
        memory_config,
        num_preferred_links,
        sub_device_id,
        use_all_gather_async_llama_sharded,
        use_optimal_ccl_for_llama,
        barrier_semaphore,
        reverse_order,
        sub_core_grid);
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
