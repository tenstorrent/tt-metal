// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_bilinear_program_factory_multicore.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_interleaved.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_sharded.hpp"

#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::pool::upsample {

static std::array<uint32_t, 4> get_input_shape(const UpsampleParams& args, const UpsampleInputs& tensor_args) {
    const Tensor& input = tensor_args.input_tensor;

    // For bilinear mode, input is the HALOED tensor, so we must use sliding_window_config for dimensions
    if (args.mode == "bilinear" && args.sliding_window_config.has_value()) {
        const sliding_window::SlidingWindowConfig& slidingWindowConfig = args.sliding_window_config.value();
        return {
            slidingWindowConfig.batch_size,
            slidingWindowConfig.input_hw.first,
            slidingWindowConfig.input_hw.second,
            slidingWindowConfig.channels};
    }  // For nearest mode use input tensor dimensions
    const Shape& input_shape = input.logical_shape();
    return {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
}

UpsampleOperation::program_factory_t UpsampleOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor_0 = tensor_args.input_tensor;
    if (args.mode == "bilinear") {
        // Bilinear is only supported for sharded inputs
        // In case of interleaved input, autosharding had previously been performed
        return program::UpsampleBilinearProgramFactory{};
    }
    if (args.mode == "nearest") {
        if (input_tensor_0.is_sharded()) {
            return program::UpsampleMultiCoreShardedProgramFactory{};
        }
        return program::UpsampleMultiCoreInterleavedProgramFactory{};
    }
    TT_THROW("Unsupported mode: only supported modes are nearest and bilinear");
}

void UpsampleOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;
    TT_FATAL(
        input_tensor_a.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");
    if (input_tensor_a.layout() == tt::tt_metal::Layout::TILE) {
        TT_FATAL(args.mode == "nearest", "Only nearest upsample mode is supported for tiled inputs");
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "Only interleaved memory layout is supported for tiled input");
        TT_FATAL(
            input_tensor_a.padded_shape() == input_tensor_a.logical_shape(),
            "Only tile aligned tile input is currently supported");
    }

    if (args.mode == "bilinear") {
        TT_FATAL(
            input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16, "Bilinear upsample requires BFLOAT16 input");
    }

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == args.output_mem_config.memory_layout(),
            "Input tensor memory layout should be same as output tensor memory layout");
        if (args.mode == "nearest") {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
                    input_tensor_a.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
                "Input tensor memory layout should be HEIGHT or BLOCK sharded");
        } else if (args.mode == "bilinear") {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
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
    const Tensor& input = tensor_args.input_tensor;
    const std::array<uint32_t, 4> input_shape = get_input_shape(args, tensor_args);

    const uint32_t out_n = input_shape[0];
    const uint32_t out_h = input_shape[1] * args.scale_factor_h;
    const uint32_t out_w = input_shape[2] * args.scale_factor_w;
    const uint32_t out_c = input_shape[3];

    const ttnn::Shape output_shape = ttnn::Shape({out_n, out_h, out_w, out_c});

    const tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::ROW_MAJOR;  // upsample only outputs row major data

    const tt::tt_metal::DataType output_data_type =
        input.dtype() == tt::tt_metal::DataType::BFLOAT8_B ? tt::tt_metal::DataType::BFLOAT16 : input.dtype();

    if (args.output_mem_config.is_sharded()) {
        TT_FATAL(
            input.memory_config().is_sharded(),
            "Output memory config is sharded but input memory config is not sharded");
        TT_FATAL(
            input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
                input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "Input memory config is not HEIGHT or BLOCK sharded");
        TT_FATAL(
            input.memory_config().shard_spec()->grid.ranges().size() == 1 ||
                input.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            "Block sharded input should have only one CoreRange");

        tt::tt_metal::ShardSpec shard_spec = args.output_mem_config.shard_spec().value();

        if (args.mode == "bilinear") {
            // Bilinear mode: Calculate output shard to handle non-exact work distribution
            // Also, cannot apply same logic as in nearest since input is halo in this case
            // meaning that the shard height is different
            const uint32_t num_cores = shard_spec.num_cores();
            const uint32_t total_output_sticks = out_n * out_h * out_w;
            const uint32_t output_sticks_padded = tt::round_up(total_output_sticks, num_cores);
            const uint32_t output_shard_height = output_sticks_padded / num_cores;
            shard_spec.shape = {output_shard_height, input.shard_spec()->shape[1]};
        } else {
            // Nearest mode: Output shard is simply input shard multiplied by scale factors
            shard_spec.shape = {
                input.shard_spec()->shape[0] * args.scale_factor_h * args.scale_factor_w, input.shard_spec()->shape[1]};
        }
        const tt::tt_metal::MemoryConfig mem_config = args.output_mem_config.with_shard_spec(shard_spec);
        return tt::tt_metal::TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(output_data_type, tt::tt_metal::PageConfig(output_layout), mem_config));
    }

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(output_data_type, tt::tt_metal::PageConfig(output_layout), args.output_mem_config));
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
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<ttnn::operations::sliding_window::SlidingWindowConfig>& sliding_window_config) {
    using OperationType = ttnn::operations::pool::upsample::UpsampleOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .scale_factor_h = scale_factor_h,
            .scale_factor_w = scale_factor_w,
            .mode = mode,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
            .sliding_window_config = sliding_window_config},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
