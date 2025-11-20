// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async_op.hpp"
#include "neighbor_pad_async_program.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "tt-metalium/tensor/tensor_utils.hpp"

namespace ttnn {

void NeighborPadAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(this->dim < 3, "Error, neighbor pad currently only supports padding non last dim, provided {}", this->dim);

    TT_FATAL(
        input_tensors[0].layout() == Layout::ROW_MAJOR,
        "Unsupported input tensor layout {}.",
        input_tensors[0].layout());

    TT_FATAL(!input_tensors[0].is_sharded(), "Neighbor pad does not support sharded input tensors.");

    TT_FATAL(padding_mode == "zeros" || padding_mode == "replicate", "Unsupported padding mode {}.", padding_mode);

    const auto& input_tensor_shape = input_tensors[0].padded_shape();
    TT_FATAL(
        padding_left <= input_tensor_shape[this->dim] && padding_right <= input_tensor_shape[this->dim],
        "One of the padding values {} or {} exceeds the shape of the input tensor in that dim {}.",
        this->padding_left,
        this->padding_right,
        input_tensor_shape[this->dim]);

    TT_FATAL(cluster_axis == 0 || cluster_axis == 1, "Unsupported cluster axis {}.", cluster_axis);

    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);
    if (this->dim > 0) {
        uint32_t outer_dim_size = 1;
        for (int d = 0; d < this->dim; d++) {
            outer_dim_size *= input_tensor_shape[d];
        }
        TT_FATAL(outer_dim_size >= this->num_links, "Not enough work to split among links, reduce num links");
    } else {
        uint32_t num_sticks_per_halo_dim = 1;
        for (int d = this->dim + 1; d < input_tensor_shape.size() - 1; d++) {
            num_sticks_per_halo_dim *= input_tensor_shape[d];
        }
        TT_FATAL(num_sticks_per_halo_dim >= this->num_links, "Not enough work to split among links, reduce num links");
    }

    if (secondary_cluster_axis.has_value()) {
        const auto& mesh_view = input_tensors[0].device()->get_view();
        uint32_t target_ring_size = (this->cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
        TT_FATAL(
            secondary_cluster_axis.value() == 0 || secondary_cluster_axis.value() == 1,
            "Unsupported secondary cluster axis {}.",
            secondary_cluster_axis.value());
        TT_FATAL(
            secondary_mesh_shape.has_value(),
            "If secondary cluster axis is specified, need to have a secondary mesh shape");
        TT_FATAL(
            !(target_ring_size % secondary_mesh_shape.value().at(0)) &&
                !(target_ring_size % secondary_mesh_shape.value().at(1)),
            "Secondary mesh shape ({},{}) is not valid given main cluster axis device count {}",
            secondary_mesh_shape.value().at(0),
            secondary_mesh_shape.value().at(1),
            target_ring_size);
    }
}

std::vector<ttnn::TensorSpec> NeighborPadAsync::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[this->dim] += (this->padding_left + this->padding_right);
    return {TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config))};
}

std::vector<Tensor> NeighborPadAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks NeighborPadAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks NeighborPadAsync::create_program_at(
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

    // cluster_axis
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

    return neighbor_pad_async_minimal(
        input_tensors[0],
        target_device,
        forward_device,
        backward_device,
        output_tensors[0],
        this->dim,
        this->padding_left,
        this->padding_right,
        this->padding_mode,
        this->final_semaphore,
        this->barrier_semaphore,
        this->num_links,
        this->topology,
        target_ring_size,
        device_index,
        this->secondary_cluster_axis,
        this->secondary_mesh_shape);
}

tt::tt_metal::operation::Hash NeighborPadAsync::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    auto input_shape = input_tensors[0].padded_shape();
    auto input_memory_layout = input_tensors[0].layout();
    auto input_dtype = input_tensors[0].dtype();
    auto input_memory_config = input_tensors[0].memory_config();
    return tt::tt_metal::operation::hash_operation<NeighborPadAsync>(
        this->dim,
        this->padding_left,
        this->padding_right,
        this->padding_mode,
        this->num_links,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        this->ring_size,
        this->secondary_cluster_axis,
        this->secondary_mesh_shape,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

}  // namespace ttnn
