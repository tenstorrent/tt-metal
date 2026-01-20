// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_to_device_operation.hpp"

#include <cstdint>

#include <tt_stl/assert.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

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
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    switch (input.layout()) {
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
        input.layout() == Layout::TILE,
        "bcast_to: Invalid tensor memory layout {}. Input tensor layout must be TILE.",
        input.layout());
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
            output->logical_shape() == operation_attributes.output_shape,
            "bcast_to: Invalid output shape {}Output shape must match operation attributes {}",
            output->logical_shape(),
            operation_attributes.output_shape);
        TT_FATAL(
            input.layout() == output->layout(),
            "bcast_to: Input {} and output {} must have same layout",
            input.layout(),
            output->layout());
        TT_FATAL(
            input.dtype() == output->dtype(),
            "bcast_to: Input {} and output {} must have same dtype",
            input.dtype(),
            output->dtype());
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
        return tensor_args.output->tensor_spec();
    }
    return TensorSpec(
        Shape{operation_attributes.output_shape},
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(),
            tt::tt_metal::PageConfig(tensor_args.input.layout()),
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
}  // namespace ttnn::operations::experimental::broadcast_to

namespace ttnn::prim {

ttnn::operations::experimental::broadcast_to::BcastToOperation::tensor_return_value_t bcast_to(
    const Tensor& input,
    const ttnn::Shape& output_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    using OperationType = ttnn::operations::experimental::broadcast_to::BcastToOperation;

    auto subtile_broadcast_type = ttnn::operations::experimental::broadcast_to::get_subtile_broadcast_type(
        input.logical_shape()[-2],
        input.logical_shape()[-1],
        output_shape[output_shape.size() - 2],
        output_shape[output_shape.size() - 1]);
    log_debug(tt::LogOp, "get_subtile_broadcast_type: {}\n", subtile_broadcast_type);

    auto operation_attributes = OperationType::operation_attributes_t{
        output_shape, memory_config.value_or(input.memory_config()), subtile_broadcast_type};
    auto tensor_args = OperationType::tensor_args_t{input, output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
