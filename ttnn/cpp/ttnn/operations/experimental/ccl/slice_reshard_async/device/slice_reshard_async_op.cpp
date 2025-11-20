// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_reshard_async_op.hpp"
#include "slice_reshard_async_program.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "tt-metalium/tensor/tensor_utils.hpp"

namespace ttnn {

void SliceReshardAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(this->dim == 0, "Error, neighbor pad currently only supports sharding dim 0, provided {}", this->dim);
    TT_FATAL(
        input_tensors[0].layout() == Layout::ROW_MAJOR,
        "Unsupported input tensor layout {}.",
        input_tensors[0].layout());

    TT_FATAL(!input_tensors[0].is_sharded(), "Slice reshard does not support sharded input tensors.");

    TT_FATAL(
        !(this->output_dim_shape % this->ring_size),
        "Output dim shape must be divisible by num devices on cluster axis");

    TT_FATAL(this->cluster_axis == 0 || this->cluster_axis == 1, "Unsupported cluster axis {}.", this->cluster_axis);

    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
}

std::vector<ttnn::TensorSpec> SliceReshardAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[this->dim] = this->output_dim_shape / this->ring_size;
    return {TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config))};
}

std::vector<Tensor> SliceReshardAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks SliceReshardAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks SliceReshardAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    const auto& mesh_view = input_tensors[0].device()->get_view();
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    devices_to_use =
        (this->cluster_axis == 0) ? mesh_view.get_devices_on_column(coord[1]) : mesh_view.get_devices_on_row(coord[0]);
    uint32_t target_ring_size = devices_to_use.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < target_ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            }
            if (i != target_ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            }
        }
    }

    return slice_reshard_async_minimal(
        input_tensors[0],
        target_device,
        forward_device,
        backward_device,
        output_tensors[0],
        this->dim,
        this->output_dim_offset,
        this->output_dim_shape,
        this->final_semaphore,
        this->barrier_semaphore,
        this->num_links,
        this->topology,
        target_ring_size,
        device_index);
}

tt::tt_metal::operation::Hash SliceReshardAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<SliceReshardAsync>(
        this->dim,
        this->output_dim_offset,
        this->output_dim_shape,
        this->num_links,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        this->ring_size,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

}  // namespace ttnn
