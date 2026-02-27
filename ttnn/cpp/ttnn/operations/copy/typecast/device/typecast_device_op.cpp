// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/hal.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
bool can_use_sharded_optimized_factory(const TypecastParams& args, const TypecastInputs& tensor_args) {
    const auto& input = tensor_args.input;
    if (!input.shard_spec().has_value()) {
        return false;
    }
    const auto& shard_spec = input.shard_spec().value();

    tt::DataFormat act_df = datatype_to_dataformat_converter(args.input_dtype);
    tt::DataFormat out_df = datatype_to_dataformat_converter(args.output_dtype);

    if (tt::tile_size(act_df) != tt::tile_size(out_df)) {
        return false;
    }

    // Sharded optimized factory requires input buffer to be in L1.
    if (input.memory_config().buffer_type() == BufferType::DRAM) {
        return false;
    }
    // Sharded optimized factory also requires the effective output buffer to be in L1.
    // If the caller configures a DRAM output, fall back to a non-optimized factory.

    if (args.output_memory_config.buffer_type() == BufferType::DRAM) {
        return false;
    }

    if (args.input_dtype != DataType::BFLOAT8_B && args.input_dtype != DataType::BFLOAT4_B) {
        if ((shard_spec.shape[1] * tt::datum_size(act_df)) % hal::get_l1_alignment() != 0) {
            return false;
        }
        size_t shard_size_in_bytes = shard_spec.shape[0] * shard_spec.shape[1] * tt::datum_size(act_df);
        if (shard_size_in_bytes % tt::tile_size(act_df) != 0) {
            return false;
        }
    }

    return !args.sub_core_grids
                .has_value();  // Typecast operation has no sub_core_grids support for optimized 2D sharded input path.
}
}  // namespace

TypecastDeviceOperation::program_factory_t TypecastDeviceOperation::select_program_factory(
    const TypecastParams& args, const TypecastInputs& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        if (can_use_sharded_optimized_factory(args, tensor_args)) {
            log_debug(tt::LogOp, "Using TypecastShardedProgramFactory");
            return TypecastShardedProgramFactory{};
        }
        log_debug(tt::LogOp, "Using TypecastProgramFactory");
        return TypecastProgramFactory{};
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

    if (input_tensor.is_sharded()) {
        const uint32_t l1_alignment = hal::get_l1_alignment();
        const uint32_t page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();
        TT_FATAL(
            page_size_bytes % l1_alignment == 0,
            "Typecast operation requires sharded input tensor page size ({} bytes) to be aligned to L1 ({} bytes)",
            page_size_bytes,
            l1_alignment);
    }

    if (preallocated_output_tensor.has_value()) {
        TT_FATAL(
            preallocated_output_tensor.value().logical_shape() == input_tensor.logical_shape(),
            "Typecast operation requires input and preallocated output logical shapes to match. Input shape: {}, "
            "Preallocated output shape: {}",
            input_tensor.logical_shape(),
            preallocated_output_tensor.value().logical_shape());
        TT_FATAL(
            preallocated_output_tensor.value().layout() == input_tensor.layout(),
            "Typecast operation requires input and output layouts to match. Input layout: {}, Output layout: {}",
            input_tensor.layout(),
            preallocated_output_tensor.value().layout());
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
