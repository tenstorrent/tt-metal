// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "tools/profiler/op_profiler.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using ttnn::operations::unary::UnaryOpType;

namespace {
void validate_supported_arch_dtype(DataType input_datatype, DataType output_datatype, UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
            TT_FATAL(
                input_datatype == DataType::INT32,
                "Unsupported input data type '{}' for UnaryOpType '{}' (Bitwise operation).",
                static_cast<int>(input_datatype),
                static_cast<int>(op_type));
            TT_FATAL(
                output_datatype == DataType::INT32,
                "Unsupported output data type '{}' for UnaryOpType '{}' (Bitwise operation).",
                static_cast<int>(output_datatype),
                static_cast<int>(op_type));
            break;
        case UnaryOpType::FMOD:
            TT_FATAL(
                (input_datatype == DataType::BFLOAT16 || input_datatype == DataType::FLOAT32),
                "Unsupported input data type '{}' for UnaryOpType '{}' (FMOD operation).",
                static_cast<int>(input_datatype),
                static_cast<int>(op_type));
            TT_FATAL(
                (output_datatype == DataType::BFLOAT16 || output_datatype == DataType::FLOAT32),
                "Unsupported output data type '{}' for UnaryOpType '{}' (FMOD operation).",
                static_cast<int>(output_datatype),
                static_cast<int>(op_type));
            break;
        default: return;
    }
}
}  // namespace

UnaryDeviceOperation::program_factory_t UnaryDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        return UnaryShardedProgramFactory{};
    }
    if (args.sub_core_grids.has_value()) {
        return UnarySubCoreGridProgramFactory{};
    }
    return UnaryProgramFactory{};
}

void UnaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void UnaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    auto out_memory_config = args.output_memory_config;
    auto output_datatype = args.output_dtype;
    if (preallocated_output_tensor.has_value()) {
        out_memory_config = preallocated_output_tensor->memory_config();
        output_datatype = preallocated_output_tensor->dtype();
    }

    auto input_datatype = input_tensor.dtype();
    for (const auto& unary_op : args.op_chain) {
        validate_supported_arch_dtype(input_datatype, output_datatype, unary_op.type());
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Unary operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to eltwise unary need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Unary operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(input_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Unary operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(input_tensor.memory_config().memory_layout()));
    }

    if (preallocated_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = preallocated_output_tensor.value().logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "When preallocted output tensor is used, Unary operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            preallocated_output_shape);

        if (!input_tensor.is_sharded()) {
            TT_FATAL(
                (preallocated_output_tensor.value().layout() == input_tensor.layout()),
                "Unary operation requires output tensor layout ({}) to match input tensor layout ({}) when working "
                "with non-sharded tensor.",
                static_cast<int>(preallocated_output_tensor.value().layout()),
                static_cast<int>(input_tensor.layout()));
        }
    }
}

TensorSpec UnaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto output_layout = tensor_args.input.layout();

    const auto output_shape = tensor_args.input.logical_shape();
    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            args.output_dtype,
            PageConfig(output_layout),
            args.output_memory_config,
            output_shape,
            tensor_args.input.padded_shape()));
}

Tensor UnaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t UnaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<UnaryDeviceOperation>(
        args,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.memory_config(),
        args.sub_core_grids,
        input_shape.volume());

    return hash;
}

bool UnaryDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

Tensor unary(
    const Tensor& input,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& preallocated_output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    auto operation_attributes = UnaryParams{
        .op_chain = op_chain,
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .preserve_fp32_precision = preserve_fp32_precision,
        .bfp8_pack_precise = bfp8_pack_precise,
        .sub_core_grids = sub_core_grids,
    };
    auto tensor_args = UnaryInputs{.input = input, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<UnaryDeviceOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
