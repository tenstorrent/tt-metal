// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::ternary {

WhereDeviceOperation::program_factory_t WhereDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.predicate.is_sharded()) {
        TT_FATAL(false, "Where sharded program factory is not implemented yet");  // Not implemented yet
        // return program::WhereShardedProgramFactory{};
    } else {
        return WhereProgramFactory{};
    }
}

void WhereDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void WhereDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& optional_output_tensor = tensor_args.optional_output_tensor;

    auto out_memory_config = args.memory_config;
    if (optional_output_tensor.has_value()) {
        out_memory_config = optional_output_tensor->memory_config();
    }

    TT_FATAL(
        predicate_tensor.storage_type() == StorageType::DEVICE,
        "Where operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(predicate_tensor.storage_type()));

    TT_FATAL(
        predicate_tensor.buffer() != nullptr,
        "Operands to eltwise where need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        predicate_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Where operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(predicate_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    if (!predicate_tensor.is_sharded()) {
        TT_FATAL(
            predicate_tensor.layout() == Layout::TILE,
            "Where operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(predicate_tensor.layout()));

        TT_FATAL(
            predicate_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Where operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(predicate_tensor.memory_config().memory_layout()));
    }

    if (optional_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto optional_output_tensor_shape = optional_output_tensor.value().logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == computed_output_shape,
            "When preallocted output tensor is used, Where operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            optional_output_tensor_shape);

        if (!predicate_tensor.is_sharded()) {
            TT_FATAL(
                (optional_output_tensor.value().layout() == Layout::TILE),
                "Where operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

TensorSpec WhereDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.memory_config.is_sharded()) {
        output_layout = tensor_args.predicate.layout();
    }

    const auto output_shape = tensor_args.predicate.logical_shape();
    return TensorSpec(output_shape, TensorLayout(args.dtype.value(), output_layout, args.memory_config));
}

Tensor WhereDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.predicate.device());
}

tt::stl::hash::hash_t WhereDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& predicate_tensor = tensor_args.predicate;
    const auto& value_true = tensor_args.value_true;
    const auto& value_false = tensor_args.value_false;
    const auto& predicate_shape = predicate_tensor.padded_shape();
    WhereVariant variant = args.where_variant;

    auto program_factory = select_program_factory(args, tensor_args);

    tt::stl::hash::hash_t hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
        args,
        program_factory.index(),
        predicate_tensor.dtype(),
        predicate_tensor.memory_config(),
        predicate_shape.volume());

    if (variant == WhereVariant::TTT) {
        hash = tt::tt_metal::operation::hash_operation<WhereDeviceOperation>(
            args,
            program_factory.index(),
            predicate_tensor.dtype(),
            predicate_tensor.memory_config(),
            value_true.dtype(),
            value_true.memory_config(),
            value_true.dtype(),
            value_true.memory_config(),
            value_false.dtype(),
            value_false.memory_config(),
            predicate_shape.volume());
    }

    return hash;
}

bool WhereDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    operation_attributes_t attributes{
        .where_variant = WhereVariant::TTT,
        .memory_config = memory_config.value_or(predicate.memory_config()),
        .input_dtype = predicate.dtype(),
        .dtype = output_dtype.value_or(value_true.dtype()),
        .compute_kernel_config = std::nullopt,
    };

    tensor_args_t args{
        .predicate = predicate,
        .value_true = value_true,
        .value_false = value_false,
        .optional_output_tensor = optional_output_tensor};

    return {attributes, args};
}

}  // namespace ttnn::operations::ternary
