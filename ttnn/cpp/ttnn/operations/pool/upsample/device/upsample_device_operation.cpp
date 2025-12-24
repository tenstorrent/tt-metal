// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_bilinear_program_factory_multicore.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_interleaved.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_sharded.hpp"

#include "tt-metalium/buffer_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::pool::upsample {
using namespace tt;
using namespace tt::tt_metal;

UpsampleOperation::program_factory_t UpsampleOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor_0 = tensor_args.input_tensor;
    if (args.mode == "bilinear") {
        // Bilinear is only supported for sharded inputs
        // In case of interleaved input, autosharding had previously been performed
        return program::UpsampleBilinearProgramFactory{};
    } else if (args.mode == "nearest") {
        if (input_tensor_0.is_sharded()) {
            return program::UpsampleMultiCoreShardedProgramFactory{};
        } else {
            return program::UpsampleMultiCoreInterleavedProgramFactory{};
        }
    } else {
        TT_THROW("Unsupported mode: only supported modes are nearest and bilinear");
    }
}

void UpsampleOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    if (input_tensor_a.layout() == Layout::TILE) {
        TT_FATAL(args.mode == "nearest", "Only nearest upsample mode is supported for tiled inputs");
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only interleaved memory layout is supported for tiled input");
        TT_FATAL(
            input_tensor_a.padded_shape() == input_tensor_a.logical_shape(),
            "Only tile aligned tile input is currently supported");
    }

    if (args.mode == "bilinear") {
        TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Bilinear upsample requires BFLOAT16 input");
    }

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == args.output_mem_config.memory_layout(),
            "Input tensor memory layout should be same as output tensor memory layout");
        if (args.mode == "nearest") {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
                    input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
                "Input tensor memory layout should be HEIGHT or BLOCK sharded");
        } else if (args.mode == "bilinear") {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Input tensor memory layout should be HEIGHT sharded");
        }
        TT_FATAL(args.mode == "bilinear" || args.mode == "nearest", "Upsample only supports bilinear or nearest mode");
        TT_FATAL(
            input_tensor_a.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
            "Input buffer should be sharded in L1");
    }
}

void UpsampleOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

UpsampleOperation::spec_return_value_t UpsampleOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& input_shape = input.logical_shape();

    const uint32_t out_n = input_shape[0];
    const uint32_t out_h = input_shape[1] * args.scale_factor_h;
    const uint32_t out_w = input_shape[2] * args.scale_factor_w;
    const uint32_t out_c = input_shape[3];

    const ttnn::Shape output_shape = ttnn::Shape({out_n, out_h, out_w, out_c});

    const Layout output_layout = Layout::ROW_MAJOR;  // upsample only outputs row major data

    const DataType output_data_type = input.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input.dtype();

    if (args.output_mem_config.is_sharded()) {
        TT_FATAL(
            input.memory_config().is_sharded(),
            "Output memory config is sharded but input memory config is not sharded");
        TT_FATAL(
            input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Input memory config is not HEIGHT or BLOCK sharded");
        TT_FATAL(
            input.memory_config().shard_spec()->grid.ranges().size() == 1 ||
                input.memory_config().memory_layout() != TensorMemoryLayout::BLOCK_SHARDED,
            "Block sharded input should have only one CoreRange");

        auto shard_spec = args.output_mem_config.shard_spec().value();
        shard_spec.shape = {
            input.shard_spec()->shape[0] * args.scale_factor_h * args.scale_factor_w, input.shard_spec()->shape[1]};
        MemoryConfig mem_config = args.output_mem_config.with_shard_spec(shard_spec);
        return TensorSpec(output_shape, TensorLayout(output_data_type, PageConfig(output_layout), mem_config));
    }

    return TensorSpec(output_shape, TensorLayout(output_data_type, PageConfig(output_layout), args.output_mem_config));
}

UpsampleOperation::tensor_return_value_t UpsampleOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::pool::upsample

namespace ttnn::prim {
ttnn::Tensor upsample(
    const ttnn::Tensor& input_tensor,
    const int scale_factor_h,
    const int scale_factor_w,
    const std::string& mode,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ttnn::operations::pool::upsample::UpsampleOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .scale_factor_h = scale_factor_h,
            .scale_factor_w = scale_factor_w,
            .mode = mode,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
