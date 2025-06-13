// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_op.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include <tt-metalium/host_api.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void AllReduceAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Error, Input tensor size should be 2 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& buffer_tensor = input_tensors[1];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");
    TT_FATAL(
        this->ring_size % 2 == 0,
        "AllReduceAsync currently only supports even number of blocks in the reduction kernel.");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(buffer_tensor.storage_type() == StorageType::DEVICE, "Operands to all_reduce need to be on device!");
    TT_FATAL(buffer_tensor.buffer() != nullptr, "Operands to all_reduce need to be allocated in buffers on device!");

    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for input tensor{}.",
        input_tensor.memory_config().memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for buffer tensor {}.",
        buffer_tensor.memory_config().memory_layout());
    TT_FATAL(
        this->output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for output tensor {}.",
        this->output_mem_config.memory_layout());

    TT_FATAL(
        buffer_tensor.memory_config().shard_spec()->grid.contains(this->output_mem_config.shard_spec()->grid),
        "The output tensor must reside on a subset of the cores of the buffer tensor");

    const uint32_t output_shard_shape_volume =
        this->output_mem_config.shard_spec()->shape[0] * this->output_mem_config.shard_spec()->shape[1];
    const uint32_t buffer_shard_shape_volume =
        buffer_tensor.memory_config().shard_spec()->shape[0] * buffer_tensor.memory_config().shard_spec()->shape[1];
    TT_FATAL(
        output_shard_shape_volume * this->ring_size <= buffer_shard_shape_volume,
        "The shard size for the buffer must be large enough to hold the intermediate tensor. Require at least {} but "
        "has {}",
        output_shard_shape_volume * this->ring_size,
        buffer_shard_shape_volume);
}

std::vector<ttnn::TensorSpec> AllReduceAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    tt::tt_metal::TensorLayout output_tensor_layout =
        tt::tt_metal::TensorLayout(this->dtype, input_tensor.tensor_spec().page_config(), this->output_mem_config);

    return {TensorSpec(shape, output_tensor_layout)};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllReduceAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
};

tt::tt_metal::operation::ProgramWithCallbacks AllReduceAsync::create_program_at(
    const ttnn::MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program is called");
    const auto mesh_view = this->mesh_device->get_view();
    std::vector<IDevice*> devices =
        (this->cluster_axis == 0) ? mesh_view.get_devices_on_column(coord[1]) : mesh_view.get_devices_on_row(coord[0]);

    IDevice* target_device =
        input_tensors[0].mesh_device() ? input_tensors[0].mesh_device()->get_device(coord) : input_tensors[0].device();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;
    for (uint32_t i = 0; i < ring_size; ++i) {
        if (devices.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(ring_size - 1);
            }
            if (i != ring_size - 1) {
                forward_device = devices.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }

    auto input_tensor_shape = input_tensors[0].padded_shape();
    auto input_tensor_buffer_layout = input_tensors[0].buffer()->buffer_layout();
    auto input_tensor_page_layout = input_tensors[0].layout();

    auto input_tensor_memory_config = input_tensors[0].memory_config();
    auto output_tensor_memory_config = output_tensors[0].memory_config();
    uint32_t input_shard_num_cores = input_tensor_memory_config.shard_spec()->grid.num_cores();
    uint32_t output_shard_num_cores = output_tensor_memory_config.shard_spec()->grid.num_cores();

    log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor_shape);
    log_debug(tt::LogOp, "input_tensor_memory_config: {}", input_tensor_memory_config);
    log_debug(tt::LogOp, "output_tensor_memory_config: {}", output_tensor_memory_config);
    log_debug(tt::LogOp, "input_shard_num_cores: {}", input_shard_num_cores);
    log_debug(tt::LogOp, "output_shard_num_cores: {}", output_shard_num_cores);
    log_debug(
        tt::LogOp,
        "input_tensor_memory_config.shard_spec()->shape: {}",
        input_tensor_memory_config.shard_spec()->shape);
    log_debug(
        tt::LogOp,
        "output_tensor_memory_config.shard_spec()->shape: {}",
        output_tensor_memory_config.shard_spec()->shape);

    log_debug(tt::LogOp, "Running TG Llama specific all_reduce_async_minimal_multi_core_with_workers");
    return all_reduce_async_minimal_multi_core_with_workers(
        input_tensors[0],
        input_tensors[1],
        target_device,
        forward_device,
        backward_device,
        output_tensors[0],
        this->dtype,
        this->num_links,
        this->ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash AllReduceAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    auto output_dtype = this->dtype;
    return tt::tt_metal::operation::hash_operation<AllReduceAsync>(
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        output_dtype);
}

namespace operations {
namespace experimental {
namespace ccl {
namespace {
Tensor all_reduce_async_impl(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    const auto mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-reduce invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    return tt::tt_metal::operation::run(
               ttnn::AllReduceAsync{
                   num_preferred_links.has_value() ? num_preferred_links.value() : 1,
                   num_devices,
                   dtype.value_or(input_tensor.dtype()),
                   memory_config.value_or(input_tensor.memory_config()),
                   topology,
                   multi_device_global_semaphore,
                   subdevice_id,
                   cluster_axis,
                   &mesh_device},
               {input_tensor, buffer_tensor})
        .at(0);
}
}  // namespace

Tensor all_reduce_async(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return all_reduce_async_impl(
        input_tensor,
        buffer_tensor,
        cluster_axis,
        *(input_tensor.mesh_device()),
        topology,
        multi_device_global_semaphore,
        dtype,
        memory_config,
        num_preferred_links,
        subdevice_id);
}

std::vector<Tensor> all_reduce_async(
    const std::vector<Tensor>& input_tensors,
    Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    std::vector<Tensor> output_tensors;
    output_tensors.reserve(input_tensors.size());
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        output_tensors.push_back(all_reduce_async_impl(
            input_tensors[i],
            buffer_tensor,
            cluster_axis,
            mesh_device,
            topology,
            multi_device_global_semaphore.global_semaphores[i],
            dtype,
            memory_config,
            num_preferred_links,
            subdevice_id));
    }
    return output_tensors;
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    const size_t num_workers_preferred = num_workers_per_link * num_links;
    if (available_cores.num_cores() < num_workers_preferred) {
        log_warning(
            tt::LogOp,
            "AllGather is being launched on a subdevice with fewer worker cores available than ideal. Ideally {} "
            "cores ({} per link and {} links) are made available but only {} are available. This may lead to "
            "performance loss.",
            num_workers_preferred,
            num_workers_per_link,
            num_links,
            available_cores.num_cores());
    }
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                sender_worker_core_range =
                    sender_worker_core_range.merge(CoreRangeSet(CoreRange(CoreCoord(x, y), CoreCoord(x, y))));
                if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                    break;
                }
            }
            if (sender_worker_core_range.num_cores() == num_workers_preferred) {
                break;
            }
        }
        if (sender_worker_core_range.num_cores() == num_workers_preferred) {
            break;
        }
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

}  // namespace ttnn
