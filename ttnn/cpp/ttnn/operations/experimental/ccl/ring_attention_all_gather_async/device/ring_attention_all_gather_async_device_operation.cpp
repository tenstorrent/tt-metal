// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async {

RingAttentionAllGatherAsyncDeviceOperation::program_factory_t
RingAttentionAllGatherAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Only one program factory available
    return RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory{};
}

void RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void RingAttentionAllGatherAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
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

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);

    TT_FATAL(
        memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout {}.",
        memory_config.memory_layout());

    // Validate output tensors if provided
    const auto& output_tensors = tensor_args.persistent_output_buffer;
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
                    output_tensor.memory_config() == operation_attributes.output_mem_config,
                    "Output tensor {} memory config should match output_mem_config",
                    i);

                // Check output tensor shape
                auto output_shape = output_tensor.logical_shape();
                auto expected_output_shape = input_shape;
                expected_output_shape[operation_attributes.dim] *= operation_attributes.ring_size;

                TT_FATAL(
                    output_shape == expected_output_shape,
                    "Output tensor {} shape mismatch. Expected shape with dimension {} scaled by ring_size {}",
                    i,
                    operation_attributes.dim,
                    operation_attributes.ring_size);
            }
        }
    }
}

RingAttentionAllGatherAsyncDeviceOperation::spec_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensor;
    const auto& input_tensor = input_tensors[0];
    auto shape = input_tensor.logical_shape();
    shape[operation_attributes.dim] *= operation_attributes.ring_size;
    std::vector<ttnn::TensorSpec> output_specs;
    output_specs.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); i++) {
        output_specs.push_back(TensorSpec(
            shape,
            TensorLayout(
                input_tensor.dtype(),
                input_tensor.tensor_spec().page_config(),
                operation_attributes.output_mem_config)));
    }
    return output_specs;
}

RingAttentionAllGatherAsyncDeviceOperation::tensor_return_value_t
RingAttentionAllGatherAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<Tensor> output_tensors;
    const auto& persistent_output_buffer = tensor_args.persistent_output_buffer;
    if (!persistent_output_buffer.empty() && persistent_output_buffer[0].has_value()) {
        output_tensors.reserve(persistent_output_buffer.size());
        for (const auto& buffer : persistent_output_buffer) {
            TT_FATAL(buffer.has_value(), "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(buffer.value());
        }
        return output_tensors;
    }
    const auto& input_tensors = tensor_args.input_tensor;
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, input_tensors[0].device()));
    }
    return output_tensors;
}

tt::stl::hash::hash_t RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash is called");

    auto subdevice_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.at(0).device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<RingAttentionAllGatherAsyncDeviceOperation>(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.topology,
        operation_attributes.cluster_axis,
        subdevice_core_range_set,
        tensor_args,
        program_factory.index());
}

std::tuple<
    RingAttentionAllGatherAsyncDeviceOperation::operation_attributes_t,
    RingAttentionAllGatherAsyncDeviceOperation::tensor_args_t>
RingAttentionAllGatherAsyncDeviceOperation::invoke(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& persistent_output_buffer,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API without 2D mesh, which is currently unsupported");
    uint32_t ring_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
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

    return {
        operation_attributes_t{
            {},
            gather_dim,
            num_links,
            ring_size,
            memory_config.value_or(input_tensors[0].memory_config()),
            topology,
            multi_device_global_semaphore,
            sub_device_id,
            cluster_axis,
        },
        tensor_args_t{.input_tensor = input_tensors, .persistent_output_buffer = optional_output_tensors}};
}

}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async
