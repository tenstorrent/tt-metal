// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_to_device_operation.hpp"

#include <cstdint>

#include "tt-metalium/assert.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::broadcast_to {
SubtileBroadcastType get_subtile_broadcast_type(uint32_t a_h, uint32_t a_w, uint32_t b_h, uint32_t b_w) {
    if (a_h == b_h && a_w == b_w) {
        return SubtileBroadcastType::NONE;
    }
    if (a_h == 1 && a_w == 1 && b_h > 1 && b_w > 1) {
        return SubtileBroadcastType::SCALAR;
    }
    if (a_h == 1 && b_h > 1) {
        return SubtileBroadcastType::ROW;
    }
    if (a_w == 1 && b_w > 1) {
        return SubtileBroadcastType::COL;
    }

    TT_THROW("Invalid subtile broadcast type");
}

BcastToOperation::program_factory_t BcastToOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    switch (input.get_layout()) {
        case Layout::TILE: return BcastToTileFactory{};
        default: TT_THROW("BcastTo: Unsupported input layout");
    }
}

void validate(
    const BcastToOperation::operation_attributes_t& operation_attributes,
    const BcastToOperation::tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(
        input.get_layout() == Layout::TILE,
        "bcast_to: Invalid tensor memory layout {}. Input tensor layout must be TILE.",
        input.get_layout());
    TT_FATAL(
        tensor_args.input.storage_type() == StorageType::DEVICE,
        "bcast_to: Invalid storage_type {}. Input tensor need to be on device",
        tensor_args.input.storage_type());
    TT_FATAL(tensor_args.input.buffer() != nullptr, "bcast_to: Input tensor need to be allocated in buffers on device");
    TT_FATAL(
        tensor_args.input.memory_config().is_sharded() == false,
        "bcast_to: Invalid input memory config {}. Input tensor sharding not supported",
        tensor_args.input.memory_config());
    TT_FATAL(
        operation_attributes.memory_config.is_sharded() == false,
        "bcast_to: Invalid output memory config {}. Output memory config sharding not supported",
        operation_attributes.memory_config);
    if (output.has_value()) {
        TT_FATAL(
            output->get_logical_shape() == operation_attributes.output_shape,
            "bcast_to: Invalid output shape {}Output shape must match operation attributes {}",
            output->get_logical_shape(),
            operation_attributes.output_shape);
        TT_FATAL(
            input.get_layout() == output->get_layout(),
            "bcast_to: Input {} and output {} must have same layout",
            input.get_layout(),
            output->get_layout());
        TT_FATAL(
            input.get_dtype() == output->get_dtype(),
            "bcast_to: Input {} and output {} must have same dtype",
            input.get_dtype(),
            output->get_dtype());
    }
}

void BcastToOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
};

void BcastToOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
};

BcastToOperation::spec_return_value_t BcastToOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->get_tensor_spec();
    }
    return TensorSpec(
        Shape{operation_attributes.output_shape},
        tt::tt_metal::TensorLayout(
            tensor_args.input.get_dtype(),
            tt::tt_metal::PageConfig(tensor_args.input.get_layout()),
            operation_attributes.memory_config));
};

BcastToOperation::tensor_return_value_t BcastToOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Let's just require it to be allocated ahead of time for now
    if (tensor_args.output.has_value()) {
        return {tensor_args.output.value()};
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

std::tuple<BcastToOperation::operation_attributes_t, BcastToOperation::tensor_args_t> BcastToOperation::invoke(
    const Tensor& input,
    const Shape& output_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    auto subtile_broadcast_type = get_subtile_broadcast_type(
        input.get_logical_shape()[-2],
        input.get_logical_shape()[-1],
        output_shape[output_shape.size() - 2],
        output_shape[output_shape.size() - 1]);
    tt::log_debug(tt::LogOp, "get_subtile_broadcast_type: {}\n", subtile_broadcast_type);
    return {
        operation_attributes_t{output_shape, memory_config.value_or(input.memory_config()), subtile_broadcast_type},
        tensor_args_t{input, output}};
}
}  // namespace ttnn::operations::experimental::broadcast_to
