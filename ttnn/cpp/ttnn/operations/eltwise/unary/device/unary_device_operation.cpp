// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::unary {

namespace {
void validate_supported_arch_dtype(tt::ARCH arch, DataType input_datatype, DataType output_datatype, UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FLOOR:
        case UnaryOpType::CEIL:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::RIGHT_SHIFT:
            TT_FATAL(arch != tt::ARCH::GRAYSKULL, fmt::format("UnaryOpType '{}' is not supported on Grayskull architecture.", static_cast<int>(op_type)));
            break;
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
            TT_FATAL(arch != tt::ARCH::GRAYSKULL, fmt::format("UnaryOpType '{}' is not supported on Grayskull architecture (Bitwise operation).", static_cast<int>(op_type)));
            TT_FATAL(input_datatype == DataType::INT32, fmt::format("Unsupported input data type '{}' for UnaryOpType '{}' (Bitwise operation).", static_cast<int>(input_datatype), static_cast<int>(op_type)));
            TT_FATAL(output_datatype == DataType::INT32, fmt::format("Unsupported output data type '{}' for UnaryOpType '{}' (Bitwise operation).", static_cast<int>(output_datatype), static_cast<int>(op_type)));
            break;
        case UnaryOpType::FMOD:
            TT_FATAL(arch != tt::ARCH::GRAYSKULL, fmt::format("UnaryOpType '{}' (FMOD operation) is not supported on Grayskull architecture.", static_cast<int>(op_type)));
            TT_FATAL(input_datatype == DataType::BFLOAT16, fmt::format("Unsupported input data type '{}' for UnaryOpType '{}' (FMOD operation).", static_cast<int>(input_datatype), static_cast<int>(op_type)));
            TT_FATAL(output_datatype == DataType::BFLOAT16, fmt::format("Unsupported output data type '{}' for UnaryOpType '{}' (FMOD operation).", static_cast<int>(output_datatype), static_cast<int>(op_type)));
            break;
        default:
            return;
    }
}
} // namespace


UnaryDeviceOperation::program_factory_t UnaryDeviceOperation::select_program_factory(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        return program::UnaryShardedProgramFactory{};
    }
    else {
        return program::UnaryProgramFactory{};
    }
}

void UnaryDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void UnaryDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args) {

    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    auto out_memory_config = args.output_memory_config;
    auto output_datatype = args.output_dtype;
    if (preallocated_output_tensor.has_value()) {
        out_memory_config = preallocated_output_tensor->memory_config();
        output_datatype = preallocated_output_tensor->get_dtype();
    }

    auto arch = input_tensor.device()->arch();
    auto input_datatype = input_tensor.get_dtype();
    for (const auto& unary_op : args.op_chain) {
        validate_supported_arch_dtype(arch, input_datatype, output_datatype, unary_op.op_type);
    }

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE,
             fmt::format("Unary operation requires input to be on Device. Input storage type: {}", static_cast<int>(input_tensor.storage_type())));

    TT_FATAL(input_tensor.buffer() != nullptr,
             "Operands to eltwise unary need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(input_tensor.memory_config().memory_layout == out_memory_config.memory_layout,
             fmt::format("Unary operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
                         static_cast<int>(input_tensor.memory_config().memory_layout),
                         static_cast<int>(out_memory_config.memory_layout)));

    if (!input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE,
                 fmt::format("Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input tensor layout: {}",
                             static_cast<int>(input_tensor.get_layout())));

        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
                 fmt::format("Unary operation requires Interleaved memory layout when working with non-sharded input tensor. Input memory layout: `{}`",
                             static_cast<int>(input_tensor.memory_config().memory_layout)));
    }

    if (preallocated_output_tensor.has_value()) {
        const auto compited_output_shape = compute_output_shapes(args, tensor_args);
        const auto preallocated_output_shape = preallocated_output_tensor->get_shape();
        TT_FATAL(preallocated_output_shape == compited_output_shape,
                 fmt::format("When preallocted output tensor is used, Unary operation requires its shape to match the computed shape. Computed shape: {}, Shape in preallocated output tensor: {}",
                             compited_output_shape, preallocated_output_shape));
    }
}

shape_return_value_t UnaryDeviceOperation::compute_output_shapes(const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return {tensor_args.input.get_shape()};
}

tensor_return_value_t UnaryDeviceOperation::create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if(tensor_args.preallocated_output.has_value()){
        return tensor_args.preallocated_output.value();
    }

    const auto output_shape = compute_output_shapes(args, tensor_args);

    auto output_layout = Layout::TILE;
    if (args.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input.get_layout();
    }

    return create_device_tensor(
        output_shape,
        args.output_dtype,
        output_layout,
        tensor_args.input.device(),
        args.output_memory_config);
}

tt::stl::hash::hash_t UnaryDeviceOperation::compute_program_hash(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.legacy_shape();

    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<UnaryDeviceOperation>(
        args,
        program_factory.index(),
        input_tensor.dtype(),
        std::get<DeviceStorage>(input_tensor.storage()).memory_config(),
        compute_volume(input_shape));

    return hash;
}

std::tuple<UnaryDeviceOperation::operation_attributes_t, UnaryDeviceOperation::tensor_args_t> UnaryDeviceOperation::invoke(
    const Tensor& input,
    const std::vector<UnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    const std::optional<Tensor>& preallocated_output)
{
    return {
        operation_attributes_t{
            .op_chain = op_chain,
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
        },
        tensor_args_t{
            .input = input,
            .preallocated_output = preallocated_output}};
}


}  // namespace ttnn::operations::unary
