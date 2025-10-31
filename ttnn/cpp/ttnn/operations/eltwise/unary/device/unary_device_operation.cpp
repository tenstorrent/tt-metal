// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tools/profiler/op_profiler.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary {

namespace {
void validate_supported_arch_dtype(
    tt::ARCH arch, DataType input_datatype, DataType output_datatype, UnaryOpType op_type) {
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
        return program::UnaryShardedProgramFactory{};
    } else {
        return program::UnaryProgramFactory{};
    }
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

    auto arch = input_tensor.device()->arch();
    auto input_datatype = input_tensor.dtype();
    for (const auto& unary_op : args.op_chain) {
        validate_supported_arch_dtype(arch, input_datatype, output_datatype, unary_op.type());
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Unary operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to eltwise unary need to be allocated in buffers on the device. Buffer is null.");

    // Allow different memory layouts for input and output when sharding is involved
    bool input_sharded = input_tensor.memory_config().is_sharded();
    bool output_sharded = out_memory_config.is_sharded();

    if (input_sharded || output_sharded) {
        // If either input or output is sharded, allow different layouts
        // But validate that both are valid (either interleaved or sharded)
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED || input_sharded,
            "Input tensor must be either interleaved or sharded. Input layout: {}",
            static_cast<int>(input_tensor.memory_config().memory_layout()));

        TT_FATAL(
            out_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED || output_sharded,
            "Output memory config must be either interleaved or sharded. Output layout: {}",
            static_cast<int>(out_memory_config.memory_layout()));
    } else {
        // If neither is sharded, layouts must match
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
            "Unary operation requires Input and Output memory layout to match when neither is sharded. Input layout: {}, Output layout: {}",
            static_cast<int>(input_tensor.memory_config().memory_layout()),
            static_cast<int>(out_memory_config.memory_layout()));
    }

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(input_tensor.layout()));

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

        if(!input_tensor.is_sharded()){
            TT_FATAL(
                (preallocated_output_tensor.value().layout() == Layout::TILE),
                "Unary operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

spec_return_value_t UnaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto output_shape = tensor_args.input.logical_shape();
    auto output_memory_config = args.output_memory_config;

    // Handle automatic shard spec generation for sharded memory configs without explicit shard specs
    if (output_memory_config.is_sharded() && !output_memory_config.shard_spec().has_value()) {
        // Get the device and compute grid
        auto device = tensor_args.input.device();
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        CoreCoord start_coord(0, 0);
        CoreCoord end_coord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1);
        CoreRangeSet core_grid({CoreRange(start_coord, end_coord)});

        // Compute automatic shard shape
        auto auto_shard_shape = ttnn::operations::unary::utils::compute_auto_shard_shape(
            output_shape,
            core_grid,
            output_memory_config.memory_layout(),
            tensor_args.input.layout());

        // Create shard spec and update memory config
        ShardSpec shard_spec(core_grid, auto_shard_shape);
        output_memory_config = MemoryConfig(
            output_memory_config.memory_layout(),
            output_memory_config.buffer_type(),
            shard_spec);
    }

    auto output_layout = Layout::TILE;
    if (output_memory_config.is_sharded()) {
        output_layout = tensor_args.input.layout();
    }

    return TensorSpec(output_shape, TensorLayout(args.output_dtype, output_layout, output_memory_config));
}

tensor_return_value_t UnaryDeviceOperation::create_output_tensors(
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
        input_tensor.memory_config(),
        input_shape.volume());

    return hash;
}

bool UnaryDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<UnaryDeviceOperation::operation_attributes_t, UnaryDeviceOperation::tensor_args_t>
UnaryDeviceOperation::invoke(
    const Tensor& input,
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& preallocated_output) {
    return {
        operation_attributes_t{
            .op_chain = op_chain,
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
            .bfp8_pack_precise = bfp8_pack_precise,
        },
        tensor_args_t{.input = input, .preallocated_output = preallocated_output}};
}

}  // namespace ttnn::operations::unary
