// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_async_op.hpp"
#include "ttnn/operations/math.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

#include <tt-metalium/host_api.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {
namespace ccl {
namespace all_reduce_detail {

AllReduceAsync create_all_reduce_async_struct(
    const Tensor& input_tensor,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const std::vector<IDevice*>& devices,
    const ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode) {
    uint32_t num_devices = devices.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    std::optional<GlobalSemaphore> semaphore = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            semaphore = semaphores.at(i);  // Get raw pointer
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(num_devices - 1);
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }

    return ttnn::AllReduceAsync{
        forward_device,
        backward_device,
        num_links,
        num_devices,
        device_index,
        memory_config.value_or(input_tensor.memory_config()),
        topology,
        semaphore.value(),
        sub_device_id,
        enable_persistent_fabric_mode};
}

}  // namespace all_reduce_detail
}  // namespace ccl

void AllReduceAsync::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(input_tensors.size() == 2, "Error, Input tensor size should be 2 but has {}", input_tensors.size());
    const auto& input_tensor = input_tensors[0];
    const auto& buffer_tensor = input_tensors[1];
    const auto& layout = input_tensors[0].get_layout();
    const auto& dtype = input_tensors[0].get_dtype();
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
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for input tensor{}.",
        input_tensor.memory_config().memory_layout);

    TT_FATAL(
        buffer_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for buffer tensor {}.",
        buffer_tensor.memory_config().memory_layout);
    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout for output tensor {}.",
        this->output_mem_config.memory_layout);

    TT_FATAL(
        buffer_tensor.memory_config().shard_spec->grid.contains(this->output_mem_config.shard_spec->grid),
        "The output tensor must reside on a subset of the cores of the buffer tensor");

    const uint32_t output_shard_shape_volume =
        this->output_mem_config.shard_spec->shape[0] * this->output_mem_config.shard_spec->shape[1];
    const uint32_t buffer_shard_shape_volume =
        buffer_tensor.memory_config().shard_spec->shape[0] * buffer_tensor.memory_config().shard_spec->shape[1];
    TT_FATAL(
        output_shard_shape_volume * this->ring_size <= buffer_shard_shape_volume,
        "The shard size for the buffer must be large enough to hold the intermediate tensor. Require at least {} but "
        "has {}",
        output_shard_shape_volume * this->ring_size,
        buffer_shard_shape_volume);
}

std::vector<ttnn::TensorSpec> AllReduceAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.get_logical_shape();
    auto output_tensor_layout =
        input_tensor.get_tensor_spec().tensor_layout().with_memory_config(this->output_mem_config);
    return {TensorSpec(shape, output_tensor_layout)};
}

tt::tt_metal::operation::ProgramWithCallbacks AllReduceAsync::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    tt::log_debug(tt::LogOp, "DEBUG: create_program is called");

    auto input_tensor_shape = input_tensors[0].get_padded_shape();
    auto input_tensor_buffer_layout = input_tensors[0].buffer()->buffer_layout();
    auto input_tensor_page_layout = input_tensors[0].layout();

    auto input_tensor_memory_config = input_tensors[0].memory_config();
    auto output_tensor_memory_config = output_tensors[0].memory_config();
    uint32_t input_shard_num_cores = input_tensor_memory_config.shard_spec->grid.num_cores();
    uint32_t output_shard_num_cores = output_tensor_memory_config.shard_spec->grid.num_cores();

    tt::log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor_shape);
    tt::log_debug(tt::LogOp, "input_tensor_memory_config: {}", input_tensor_memory_config);
    tt::log_debug(tt::LogOp, "output_tensor_memory_config: {}", output_tensor_memory_config);
    tt::log_debug(tt::LogOp, "input_shard_num_cores: {}", input_shard_num_cores);
    tt::log_debug(tt::LogOp, "output_shard_num_cores: {}", output_shard_num_cores);
    tt::log_debug(
        tt::LogOp, "input_tensor_memory_config.shard_spec->shape: {}", input_tensor_memory_config.shard_spec->shape);
    tt::log_debug(
        tt::LogOp, "output_tensor_memory_config.shard_spec->shape: {}", output_tensor_memory_config.shard_spec->shape);

    tt::log_debug(tt::LogOp, "Running TG Llama specific all_reduce_async_minimal_multi_core_with_workers");
    return all_reduce_async_minimal_multi_core_with_workers(
        input_tensors[0],
        input_tensors[1],
        this->forward_device,
        this->backward_device,
        output_tensors[0],
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->topology,
        this->semaphore,
        this->sub_device_id,
        this->enable_persistent_fabric_mode);
}

const tt::tt_metal::operation::Hash AllReduceAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].get_padded_shape();
    auto input_memory_layout = input_tensors[0].get_layout();
    auto input_dtype = input_tensors[0].get_dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<AllReduceAsync>(
        this->num_links,
        this->ring_size,
        this->ring_index,
        this->output_mem_config,
        this->topology,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_reduce_async(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<size_t> num_preferred_links,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    bool enable_persistent_fabric_mode) {
    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;

    tt::tt_metal::operation::launch_op(
        [num_preferred_links,
         memory_config,
         mesh_view,
         cluster_axis,
         num_devices,
         topology,
         semaphores,
         subdevice_id,
         enable_persistent_fabric_mode](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_device_tensor = input_tensors.at(0);

            TT_FATAL(
                mesh_view.is_mesh_2d(),
                "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
            const auto coordinate = mesh_view.find_device(input_device_tensor.device()->id());
            std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate[1])
                                                                : mesh_view.get_devices_on_row(coordinate[0]);

            const auto& input_tensor = input_tensors.at(0);
            const auto& buffer_tensor = input_tensors.at(1);

            return tt::tt_metal::operation::run(
                ttnn::ccl::all_reduce_detail::create_all_reduce_async_struct(
                    input_device_tensor,
                    num_preferred_links.has_value() ? num_preferred_links.value() : 1,
                    memory_config,
                    devices,
                    topology,
                    semaphores,
                    subdevice_id,
                    enable_persistent_fabric_mode),
                {input_tensor, buffer_tensor});
        },
        {input_tensor, buffer_tensor},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, bool persistent_fabric_mode, const CoreRangeSet& available_cores) {
    std::tuple<CoreRangeSet, std::vector<CoreCoord>> result;
    CoreRangeSet sender_worker_core_range;
    if (persistent_fabric_mode) {
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
    } else {
        sender_worker_core_range =
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_workers_per_link - 1, num_links - 1)));
    }
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

}  // namespace ttnn
