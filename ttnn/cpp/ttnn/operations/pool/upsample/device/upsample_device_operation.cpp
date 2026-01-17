// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_bilinear_program_factory_multicore.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_interleaved.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_sharded.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_nearest_float_program_factory.hpp"
#include "upsample/device/upsample_device_operation_types.hpp"

#include <cmath>

namespace ttnn::operations::pool::upsample {

static std::array<uint32_t, 4> get_input_shape(const UpsampleParams& args, const UpsampleInputs& tensor_args) {
    // For bilinear mode, input is haloed - use sliding_window_config for original dimensions
    if (args.mode == "bilinear" && args.sliding_window_config.has_value()) {
        const auto& swc = args.sliding_window_config.value();
        return {swc.batch_size, swc.input_hw.first, swc.input_hw.second, swc.channels};
    }
    const auto& s = tensor_args.input_tensor.logical_shape();
    return {s[0], s[1], s[2], s[3]};
}

UpsampleOperation::program_factory_t UpsampleOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input_tensor;

    if (args.mode == "bilinear") {
        return program::UpsampleBilinearProgramFactory{};
    }

    UpsamplePath path = select_upsample_path(input, args.scale_factor_h, args.scale_factor_w, args.mode);

    if (path == UpsamplePath::INTEGER_OPTIMIZED) {
        if (input.is_sharded()) {
            return program::UpsampleMultiCoreShardedProgramFactory{};
        }
        return program::UpsampleMultiCoreInterleavedProgramFactory{};
    }

    return program::UpsampleNearestFloatProgramFactory{};
}

void UpsampleOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;

    // Basic tensor validation
    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input tensor must have allocated buffer");

    // Scale factor validation
    TT_FATAL(args.scale_factor_h > 0.0f, "scale_factor_h must be positive, got {}", args.scale_factor_h);
    TT_FATAL(args.scale_factor_w > 0.0f, "scale_factor_w must be positive, got {}", args.scale_factor_w);

    // Mode validation
    TT_FATAL(args.mode == "bilinear" || args.mode == "nearest", "mode must be 'nearest' or 'bilinear'");

    // Bilinear-specific requirements
    if (args.mode == "bilinear") {
        TT_FATAL(
            is_integer_scale(args.scale_factor_h) && is_integer_scale(args.scale_factor_w),
            "bilinear mode requires integer scale factors, got ({}, {})",
            args.scale_factor_h,
            args.scale_factor_w);
        TT_FATAL(input.dtype() == tt::tt_metal::DataType::BFLOAT16, "Bilinear upsample requires BFLOAT16 input");
        TT_FATAL(input.memory_config().is_sharded(), "Bilinear upsample requires sharded input tensor");
        TT_FATAL(
            input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "Bilinear upsample requires HEIGHT_SHARDED input tensor");
    }

    // TILE layout requirements
    if (input.layout() == tt::tt_metal::Layout::TILE) {
        TT_FATAL(
            input.padded_shape() == input.logical_shape(), "Tiled input must be tile-aligned (no padding difference)");
    }

    // Path validation with detailed error message
    UpsamplePath path = select_upsample_path(input, args.scale_factor_h, args.scale_factor_w, args.mode);
    TT_FATAL(
        path != UpsamplePath::UNSUPPORTED,
        "{}",
        generate_unsupported_config_message(input, args.scale_factor_h, args.scale_factor_w, args.mode));
}

void UpsampleOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

UpsampleOperation::spec_return_value_t UpsampleOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input_tensor;
    const auto input_shape = get_input_shape(args, tensor_args);

    // Compute output dimensions (floor for PyTorch compatibility)
    const uint32_t out_n = input_shape[0];
    const uint32_t out_h = static_cast<uint32_t>(std::floor(input_shape[1] * args.scale_factor_h));
    const uint32_t out_w = static_cast<uint32_t>(std::floor(input_shape[2] * args.scale_factor_w));
    const uint32_t out_c = input_shape[3];
    const ttnn::Shape output_shape({out_n, out_h, out_w, out_c});

    constexpr auto output_layout = tt::tt_metal::Layout::ROW_MAJOR;
    const auto output_dtype =
        input.dtype() == tt::tt_metal::DataType::BFLOAT8_B ? tt::tt_metal::DataType::BFLOAT16 : input.dtype();

    auto make_spec = [&](const tt::tt_metal::MemoryConfig& mc) {
        return tt::tt_metal::TensorSpec(
            output_shape, tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(output_layout), mc));
    };

    const auto& input_mc = input.memory_config();

    // ND sharded → float path (always)
    if (input_mc.is_sharded() && input_mc.created_with_nd_shard_spec()) {
        return make_spec(compute_nd_output_mem_config(input_mc, args.scale_factor_h, args.scale_factor_w));
    }

    // Non-sharded output
    if (!args.output_mem_config.is_sharded()) {
        return make_spec(args.output_mem_config);
    }

    // Sharded output - need input to be sharded too
    TT_FATAL(input_mc.is_sharded(), "Output memory config is sharded but input memory config is not sharded");

    // Determine path and compute output mem config accordingly
    UpsamplePath path = select_upsample_path(input, args.scale_factor_h, args.scale_factor_w, args.mode);

    if (path == UpsamplePath::INTEGER_OPTIMIZED) {
        return make_spec(compute_integer_output_mem_config(
            args.output_mem_config, input, args.mode, args.scale_factor_h, args.scale_factor_w, out_n, out_h, out_w));
    }

    return make_spec(compute_float_output_mem_config(input_mc, out_n, out_h, out_w));
}

UpsampleOperation::tensor_return_value_t UpsampleOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::pool::upsample

namespace ttnn::prim {
ttnn::Tensor upsample(
    const ttnn::Tensor& input_tensor,
    const float scale_factor_h,
    const float scale_factor_w,
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
