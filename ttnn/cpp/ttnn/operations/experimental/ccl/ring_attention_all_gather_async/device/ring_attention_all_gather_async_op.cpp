// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_op.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn {

void RingAttentionAllGatherAsync::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_FATAL(
        !input_tensors.empty(), "Error, Input tensor size should be greater than 0 but has {}", input_tensors.size());

    const auto& first_input_tensor = input_tensors[0];
    const auto& dtype = first_input_tensor.dtype();
    const auto& memory_config = first_input_tensor.memory_config();
    const auto& input_shape = first_input_tensor.logical_shape();

    // Validate all input tensors
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const auto& input_tensor = input_tensors[i];

        TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor {} must be tiled", i);
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor {} must be on device", i);
        TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor {} must be allocated in buffers on device", i);

        TT_FATAL(
            input_tensor.dtype() == dtype,
            "All input tensors must have the same dtype. Input tensor {} has dtype {} but expected {}",
            i,
            input_tensor.dtype(),
            dtype);

        TT_FATAL(
            input_tensor.memory_config() == memory_config,
            "All input tensors must have the same memory config. Input tensor {} has different memory config",
            i);

        TT_FATAL(
            input_tensor.logical_shape() == input_shape,
            "All input tensors must have the same shape. Input tensor {} has different shape",
            i);
    }

    TT_FATAL(this->num_links > 0, "Error, num_links should be more than 0 but has {}", this->num_links);

    TT_FATAL(
        memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout {}.",
        memory_config.memory_layout());

    // Validate output tensors if provided
    if (!output_tensors.empty()) {
        TT_FATAL(
            output_tensors.size() == input_tensors.size(),
            "Number of output tensors ({}) must match number of input tensors ({})",
            output_tensors.size(),
            input_tensors.size());

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            if (output_tensors[i].has_value()) {
                const auto& output_tensor = output_tensors[i].value();

                TT_FATAL(output_tensor.layout() == Layout::TILE, "Output tensor {} must be tiled", i);
                TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor {} must be on device", i);

                TT_FATAL(
                    output_tensor.dtype() == dtype,
                    "Output tensor {} dtype should match input tensors but has {}",
                    i,
                    output_tensor.dtype());

                TT_FATAL(
                    output_tensor.memory_config() == this->output_mem_config,
                    "Output tensor {} memory config should match output_mem_config",
                    i);

                // Check output tensor shape
                auto output_shape = output_tensor.logical_shape();
                auto expected_output_shape = input_shape;
                expected_output_shape[this->dim] *= this->ring_size;

                TT_FATAL(
                    output_shape == expected_output_shape,
                    "Output tensor {} shape mismatch. Expected shape with dimension {} scaled by ring_size {}",
                    i,
                    this->dim,
                    this->ring_size);
            }
        }
    }
}

std::vector<ttnn::TensorSpec> RingAttentionAllGatherAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[this->dim] *= this->ring_size;
    std::vector<ttnn::TensorSpec> output_specs;
    output_specs.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); i++) {
        output_specs.push_back(TensorSpec(
            shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), output_mem_config)));
    }
    return output_specs;
}

std::vector<Tensor> RingAttentionAllGatherAsync::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return tt::tt_metal::operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks RingAttentionAllGatherAsync::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks RingAttentionAllGatherAsync::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto* mesh_device = input_tensors[0].device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    const auto& mesh_view = input_tensors[0].device()->get_view();
    devices_to_use = (this->cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(coord[1])
                                                       : mesh_view.get_devices_on_row(coord[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->ring_size - 1);
            }
            if (i != this->ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    return ring_attention_all_gather_async_multi_core_with_workers(
        input_tensors,
        target_device,
        forward_device,
        backward_device,
        output_tensors,
        this->dim,
        this->num_links,
        this->ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->sub_device_id);
}

tt::tt_metal::operation::Hash RingAttentionAllGatherAsync::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    log_trace(tt::LogOp, "RingAttentionAllGatherAsync::compute_program_hash is called");

    const ttnn::Tensor& input_tensor = input_tensors[0];

    auto subdevice_id = this->sub_device_id;
    auto* mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    return tt::tt_metal::operation::hash_operation<RingAttentionAllGatherAsync>(
        this->dim,
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        subdevice_core_range_set,
        input_tensor);
}

namespace operations::experimental::ccl {

namespace {

std::vector<Tensor> ring_attention_all_gather_async_impl(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API withou 2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensors[0].logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    std::vector<std::optional<Tensor>> optional_output_tensors;
    optional_output_tensors.reserve(persistent_output_buffer.size());
    for (const auto& buffer : persistent_output_buffer) {
        optional_output_tensors.push_back(buffer);
    }

    return tt::tt_metal::operation::run(
        ttnn::RingAttentionAllGatherAsync{
            {},
            gather_dim,
            num_links,
            num_devices,
            memory_config.value_or(input_tensors[0].memory_config()),
            topology,
            multi_device_global_semaphore,
            sub_device_id,
            cluster_axis},
        input_tensors,
        {},
        optional_output_tensors);
}
}  // namespace

std::vector<Tensor> ring_attention_all_gather_async(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    return ring_attention_all_gather_async_impl(
        input_tensors,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        memory_config,
        topology,
        sub_device_id);
}

}  // namespace operations::experimental::ccl

}  // namespace ttnn
