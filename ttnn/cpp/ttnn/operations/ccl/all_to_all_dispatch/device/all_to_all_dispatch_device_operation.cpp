// SPDX-FileCopyrightText: © Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "all_to_all_dispatch_device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ccl {

AllToAllDispatchDeviceOperation::program_factory_t AllToAllDispatchDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return AllToAllDispatchSparse{};
}

void AllToAllDispatchDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;

    TT_FATAL(input_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR, "Input tensor must be in row major layout");
    TT_FATAL(
        indices_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR, "Indices tensor must be in row major layout");

    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16, "Input tensor must be bfloat16");
    TT_FATAL(indices_tensor.get_dtype() == tt::tt_metal::DataType::UINT16, "Indices tensor must be uint32");
    TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Output memory config must not be sharded");

    auto output_specs = compute_output_specs(operation_attributes, tensor_args);

    if (tensor_args.optional_output_tensors.has_value()) {
        auto output_tensors = tensor_args.optional_output_tensors.value();
        auto sparse_token_tensor = output_tensors[0];
        auto metadata_tensor = output_tensors[1];
        TT_FATAL(
            sparse_token_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Output tensor must be in row major layout");
        TT_FATAL(
            metadata_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "Output metadata tensor must be in row major layout");

        TT_FATAL(
            output_specs[0] == sparse_token_tensor.get_tensor_spec(),
            "Optional sparse output token tensor spec {} does not match computed output spec {}",
            sparse_token_tensor.get_tensor_spec(),
            output_specs[0]);
        TT_FATAL(
            output_specs[1] == metadata_tensor.get_tensor_spec(),
            "Optional metadata tensor spec {} does not match computed output spec {}",
            metadata_tensor.get_tensor_spec(),
            output_specs[1]);
    }

    auto input_shape = input_tensor.get_tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.get_tensor_spec().logical_shape();
    TT_FATAL(
        input_shape.rank() == 4 && (input_shape.rank() == indices_shape.rank()),
        "Input and indices tensor must have the same number of dimensions");
    for (uint32_t i = 0; i < indices_shape.rank() - 1; i++) {
        TT_FATAL(
            input_shape[i] == indices_shape[i],
            "Input and indices tensor must have the same shape for all dimensions except the last. Mismatch at "
            "dimension {} with shape {} and {}",
            i,
            input_shape[i],
            indices_shape[i]);
    }

    TT_FATAL(operation_attributes.axis.has_value(), "Axis must be specified at the moment");
    TT_FATAL(
        operation_attributes.cross_device_semaphore.has_value(),
        "Cross device semaphore must be specified at the moment");

    TT_FATAL(operation_attributes.num_links == 1, "Number of links must be 1, got {}", operation_attributes.num_links);

    TT_FATAL(operation_attributes.topology == tt::tt_fabric::Topology::Linear, "Topology must be linear at the moment");
}

void AllToAllDispatchDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

AllToAllDispatchDeviceOperation::spec_return_value_t AllToAllDispatchDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.get_tensor_spec().logical_shape();
    auto indices_shape = tensor_args.expert_indices_tensor.get_tensor_spec().logical_shape();
    auto mapping_shape = tensor_args.expert_mapping_tensor.get_tensor_spec().logical_shape();

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();

    // experts are expert parallel across devices
    // tokens are data parallel across devices
    // when axis is specified, we assume that tokens are only data parallel across the specified axis, and duplicated
    // along the other axis the indices match the token tensor the mapping tensor maps the experts to where they are on
    // the device mesh the mapping tensor is generally the same for all devices, except for the case where we have a
    // shared expert in that case, we can hide the fact that the expert is also on the other devices by setting the
    // mapping tensor to 0 for all other devices if axis is specified, we only route the tokens along the specified
    // axis, and skip any experts that are not on the specified axis

    uint32_t dispatch_devices = mesh_view.num_devices();
    uint32_t tokens_per_device = input_shape[0];
    uint32_t hidden_size = input_shape[-1];
    if (operation_attributes.axis.has_value()) {
        uint32_t axis = operation_attributes.axis.value();
        log_debug(tt::LogOp, "axis: {}", axis);
        dispatch_devices = axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    }

    // final batch in the metadata tensor
    uint32_t dispatched_tokens = tokens_per_device * dispatch_devices;
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t seq_len = indices_shape[-2];

    auto output_shape = ttnn::Shape({1, dispatched_tokens, seq_len, hidden_size});
    auto metadata_shape = ttnn::Shape({1, dispatched_tokens, seq_len, selected_experts_k});

    log_debug(tt::LogOp, "output_shape: {}", output_shape);
    log_debug(tt::LogOp, "metadata_shape: {}", metadata_shape);
    log_debug(tt::LogOp, "input_tensor_shape: {}", input_shape);
    log_debug(tt::LogOp, "indices_shape: {}", indices_shape);
    log_debug(tt::LogOp, "mapping_shape: {}", mapping_shape);
    log_debug(tt::LogOp, "dispatch_devices: {}", dispatch_devices);
    log_debug(tt::LogOp, "hidden_size: {}", hidden_size);
    log_debug(tt::LogOp, "dispatched_tokens: {}", dispatched_tokens);
    log_debug(tt::LogOp, "selected_experts_k: {}", selected_experts_k);

    auto mem_config = operation_attributes.output_mem_config;
    auto output_tokens_spec = TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), mem_config));
    auto metadata_spec = TensorSpec(
        Shape(metadata_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.expert_indices_tensor.get_dtype(),
            tt::tt_metal::PageConfig(tensor_args.expert_indices_tensor.get_layout()),
            mem_config));
    if (tensor_args.optional_output_tensors.has_value()) {
        auto output_tensors = tensor_args.optional_output_tensors.value();
        auto preallocated_output_spec = output_tensors[0].get_tensor_spec();
        auto preallocated_metadata_spec = output_tensors[1].get_tensor_spec();
        return {preallocated_output_spec, preallocated_metadata_spec};
    }
    return {output_tokens_spec, metadata_spec};
}

AllToAllDispatchDeviceOperation::tensor_return_value_t AllToAllDispatchDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensors.has_value()) {
        return tensor_args.optional_output_tensors.value();
    }
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto output_tensor = create_device_tensor(output_spec[0], tensor_args.input_tensor.device());
    auto metadata_tensor = create_device_tensor(output_spec[1], tensor_args.input_tensor.device());
    return {output_tensor, metadata_tensor};
}

std::tuple<AllToAllDispatchDeviceOperation::operation_attributes_t, AllToAllDispatchDeviceOperation::tensor_args_t>
AllToAllDispatchDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    tt::tt_metal::SubDeviceId subdevice_id,
    const std::optional<GlobalSemaphore>& global_semaphore) {
    return {
        operation_attributes_t{
            .subdevice_id = std::move(subdevice_id),
            .output_mem_config = memory_config,
            .axis = axis,
            .num_links = num_links,
            .topology = topology,
            .cross_device_semaphore = global_semaphore},
        tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices_tensor = expert_indices_tensor,
            .expert_mapping_tensor = expert_mapping_tensor,
            .optional_output_tensors = optional_output_tensors}};
}

}  // namespace ttnn::operations::ccl
