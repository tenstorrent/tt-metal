// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void AllGatherConcat::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    const auto& layout = input_tensors[0].layout();
    const auto& dtype = input_tensors[0].dtype();
    const auto& page_size = input_tensors[0].buffer()->page_size();
    const auto input_core_ranges = input_tensor.buffer()->shard_spec().grid().ranges();
    const auto padded_input_shape = input_tensor.padded_shape();
    TT_FATAL(page_size % input_tensors[0].buffer()->alignment() == 0, "All Gather currently requires aligned pages");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to all_gather need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to all_gather need to be allocated in buffers on device!");
    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    TT_FATAL(
        this->num_links <= input_tensor.device()->compute_with_storage_grid_size().y,
        "Worker cores used by links are parallelizaed over rows");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "Unsupported memory layout {}.",
        input_tensor.memory_config().memory_layout());

    TT_FATAL(
        input_core_ranges[0].start_coord.x == 1 && input_core_ranges[0].end_coord.x == 3 &&
            input_core_ranges[0].start_coord.y == 0 && input_core_ranges[0].end_coord.y == 1 &&
            input_core_ranges[1].start_coord.x == 1 && input_core_ranges[1].end_coord.x == 2 &&
            input_core_ranges[1].start_coord.y == 2 && input_core_ranges[1].end_coord.y == 2,
        "Unsupported input core ranges!");
    CoreCoord grid_size = input_tensors[0].device()->compute_with_storage_grid_size();
    TT_FATAL(grid_size.x >= 3 && grid_size.y >= 3, "Input core grid out of bound!");
    TT_FATAL(
        padded_input_shape[0] == 1 && padded_input_shape[1] == 8 && padded_input_shape[3] == 128,
        "Unsupported input shape, should be [1, 8, 32, 128] or [1, 8, 8, 128]!");
}

std::vector<ttnn::TensorSpec> AllGatherConcat::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto input_shape = input_tensor.padded_shape();  // TODO: Replace with logical_shape()
    auto num_heads = this->num_heads;
    auto sequence_length = input_shape[0];
    auto batch = input_shape[1];
    auto head_dim = input_shape[3];
    // pad batch to 32 if necessary
    uint32_t batch_size = 32;
    if (batch < batch_size) {
        batch = batch_size;
    }
    auto hidden_dim = num_heads * head_dim;

    Shape output_shape({sequence_length, 1, batch, hidden_dim});
    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::Layout::TILE, this->output_mem_config))};
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGatherConcat::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherConcat::create_program_at(
    const ttnn::MeshCoordinate& mesh_coord,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program is called");

    const auto& input_tensor = input_tensors[0];
    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(), "all-gather invoked with cluster_axis API on >2D mesh, which is currently unsupported");
    const auto target_device = mesh_device->get_device(mesh_coord);
    std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(mesh_coord[1])
                                                        : mesh_view.get_devices_on_row(mesh_coord[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->ring_size; ++i) {
        if (devices.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (this->topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(this->ring_size - 1);
            }
            if (i != this->ring_size - 1) {
                forward_device = devices.at(i + 1);
            } else if (this->topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }

    log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_concat_llama_sharded is called");
    CoreCoord compute_with_storage_grid_size = input_tensors[0].device()->compute_with_storage_grid_size();
    return all_gather_concat_llama_sharded(
        input_tensors[0],
        input_tensors[1],
        target_device,
        forward_device,
        backward_device,
        output_tensors[0],
        this->dim,
        this->num_links,
        this->ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->sub_device_id,
        this->num_heads);
}

tt::tt_metal::operation::Hash AllGatherConcat::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();

    return tt::tt_metal::operation::hash_operation<AllGatherConcat>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        this->num_heads);
}

namespace operations {
namespace experimental {
namespace ccl {

Tensor all_gather_concat(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    const uint32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& global_semaphore,
    const uint32_t num_heads,
    const MemoryConfig& memory_config,
    const std::optional<uint32_t> num_links,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto mesh_view = mesh_device.get_view();
    uint32_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    int32_t rank = input_tensor.logical_shape().rank();

    int32_t gather_dim = (dim < 0) ? rank + dim : dim;
    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    return tt::tt_metal::operation::run(
               ttnn::AllGatherConcat{
                   gather_dim,
                   num_links.value_or(1),
                   num_devices,
                   memory_config,
                   topology,
                   global_semaphore,
                   sub_device_id,
                   num_heads,
                   cluster_axis},
               {input_tensor, buffer_tensor})
        .at(0);
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
