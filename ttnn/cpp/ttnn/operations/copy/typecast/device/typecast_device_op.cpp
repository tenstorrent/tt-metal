// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::copy {

TypecastDeviceOperation::program_factory_t TypecastDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        return program::TypecastShardedProgramFactory{};
    }
    if (args.sub_core_grids.has_value()) {
        return program::TypecastSubgridProgramFactory{};
    }
    return program::TypecastProgramFactory{};
}

void TypecastDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TypecastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
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

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Typecast operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(input_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "Typecast operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(input_tensor.layout()));

        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Typecast operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(input_tensor.memory_config().memory_layout()));
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
                (preallocated_output_tensor.value().layout() == Layout::TILE),
                "Typecast operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

spec_return_value_t TypecastDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input.layout();
    }

    const auto output_shape = tensor_args.input.logical_shape();
    return TensorSpec(output_shape, TensorLayout(args.output_dtype, output_layout, args.output_memory_config));
}

tensor_return_value_t TypecastDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return tt::tt_metal::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t TypecastDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<TypecastDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_tensor.memory_config(), input_shape.volume());

    return hash;
}

bool TypecastDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::copy

namespace ttnn::prim {
ttnn::operations::copy::TypecastDeviceOperation::tensor_return_value_t typecast(
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& preallocated_output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::copy::TypecastDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .input_dtype = input.dtype(),
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
            .bfp8_pack_precise = bfp8_pack_precise,
            .sub_core_grids = sub_core_grids,
        },
        OperationType::tensor_args_t{.input = input, .preallocated_output = preallocated_output});
}
}  // namespace ttnn::prim
