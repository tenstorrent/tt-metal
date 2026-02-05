// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

TypecastDeviceOperation::program_factory_t TypecastDeviceOperation::select_program_factory(
    const TypecastParams& args, const TypecastInputs& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        log_debug(tt::LogOp, "Using TypecastShardedProgramFactory");
        return TypecastShardedProgramFactory{};
    }
    if (args.sub_core_grids.has_value()) {
        log_debug(tt::LogOp, "Using TypecastSubgridProgramFactory");
        return TypecastSubgridProgramFactory{};
    }

    if (tensor_args.input.layout() == Layout::ROW_MAJOR) {
        log_debug(tt::LogOp, "Using TypecastRowMajorChunkedProgramFactory");
        return TypecastRowMajorChunkedProgramFactory{};
    }

    log_debug(tt::LogOp, "Using TypecastProgramFactory");
    return TypecastProgramFactory{};
}

void TypecastDeviceOperation::validate_on_program_cache_hit(
    const TypecastParams& args, const TypecastInputs& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TypecastDeviceOperation::validate_on_program_cache_miss(
    const TypecastParams& args, const TypecastInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    auto out_memory_config = args.output_memory_config;
    if (preallocated_output_tensor.has_value()) {
        out_memory_config = preallocated_output_tensor->memory_config();
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Typecast operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to Typecast need to be allocated in buffers on the device. Buffer is null.");

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            args.sub_core_grids.has_value() == false,
            "Typecast operation does not support sub_core_grids when input tensor is in Row-Major layout.");
        TT_FATAL(
            input_tensor.padded_shape()[-1] % 32 == 0,
            "Typecast operation requires Row-Major input tensor's padded shape to be multiple of 32. "
            "Padded shape: {}",
            input_tensor.padded_shape());
    }

    const TensorMemoryLayout& input_tensor_memory_layout = input_tensor.memory_config().memory_layout();
    TT_FATAL(
        input_tensor_memory_layout == out_memory_config.memory_layout(),
        "Typecast operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        input_tensor_memory_layout,
        out_memory_config.memory_layout());

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor_memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Typecast operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            input_tensor_memory_layout);
    } else {
        TT_FATAL(
            !args.sub_core_grids.has_value(),
            "Typecast operation has sub_core_grids support for non-sharded inputs only");
    }

    if (preallocated_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = preallocated_output_tensor.value().logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "When preallocted output tensor is used, Typecast operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            preallocated_output_shape);

        if (!input_tensor.is_sharded()) {
            TT_FATAL(
                preallocated_output_tensor.value().layout() == input_tensor.layout(),
                "Typecast operation requires input and output layouts to match. Input layout: {}, Output layout: {}",
                input_tensor.layout(),
                preallocated_output_tensor.value().layout());
        }
    }
}

TensorSpec TypecastDeviceOperation::compute_output_specs(
    const TypecastParams& args, const TypecastInputs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const Layout output_layout = tensor_args.input.layout();

    const Shape output_shape = tensor_args.input.logical_shape();
    return TensorSpec(output_shape, TensorLayout(args.output_dtype, output_layout, args.output_memory_config));
}

Tensor TypecastDeviceOperation::create_output_tensors(const TypecastParams& args, const TypecastInputs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return tt::tt_metal::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t TypecastDeviceOperation::compute_program_hash(
    const TypecastParams& args, const TypecastInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    auto program_factory = select_program_factory(args, tensor_args);

    operation::Hash hash;

    // For tile layout, only volume matters. For row-major, actual shape dimensions matter.
    if (input_tensor.layout() == Layout::TILE) {
        hash = operation::hash_operation<TypecastDeviceOperation>(
            args,
            program_factory.index(),
            input_tensor.dtype(),
            input_tensor.memory_config(),
            input_shape.volume(),
            input_tensor.layout());
    } else {
        hash = operation::hash_operation<TypecastDeviceOperation>(
            args,
            program_factory.index(),
            input_tensor.dtype(),
            input_tensor.memory_config(),
            input_shape,
            input_tensor.layout());
    }
    return hash;
}

bool TypecastDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::prim

namespace ttnn::prim {
Tensor typecast(
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& preallocated_output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::device_operation::launch<TypecastDeviceOperation>(
        TypecastParams{
            .input_dtype = input.dtype(),
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
            .bfp8_pack_precise = bfp8_pack_precise,
            .sub_core_grids = sub_core_grids,
        },
        TypecastInputs{.input = input, .preallocated_output = preallocated_output});
}
}  // namespace ttnn::prim
