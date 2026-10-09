// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/upsample/device/upsample_device_operation.hpp"

#include <cmath>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim::qsr {

namespace upsample_common = ttnn::operations::pool::upsample;

UpsampleOperation::program_factory_t UpsampleOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const Tensor& input) {
    if (input.is_sharded()) {
        return UpsampleMultiCoreShardedProgramFactory{};
    }
    return UpsampleMultiCoreInterleavedProgramFactory{};
}

void UpsampleOperation::validate_on_program_cache_miss(const operation_attributes_t& args, const Tensor& input) {
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input tensor must have allocated buffer");
    TT_FATAL(args.scale_factor_h > 0.0f, "scale_factor_h must be positive, got {}", args.scale_factor_h);
    TT_FATAL(args.scale_factor_w > 0.0f, "scale_factor_w must be positive, got {}", args.scale_factor_w);
    TT_FATAL(
        args.mode == "nearest",
        "ttnn.experimental.quasar.upsample: only mode='nearest' is ported, got '{}'",
        args.mode);
    TT_FATAL(
        upsample_common::is_integer_scale(args.scale_factor_h) &&
            upsample_common::is_integer_scale(args.scale_factor_w),
        "ttnn.experimental.quasar.upsample: only integer scale factors are ported, got ({}, {})",
        args.scale_factor_h,
        args.scale_factor_w);
    if (input.layout() == tt::tt_metal::Layout::TILE) {
        TT_FATAL(
            input.padded_shape() == input.logical_shape(), "Tiled input must be tile-aligned (no padding difference)");
    }
    const upsample_common::UpsamplePath path =
        upsample_common::select_upsample_path(input, args.scale_factor_h, args.scale_factor_w, args.mode);
    TT_FATAL(
        path == upsample_common::UpsamplePath::INTEGER_OPTIMIZED,
        "ttnn.experimental.quasar.upsample: unsupported configuration (only the integer nearest path is ported). {}",
        upsample_common::generate_unsupported_config_message(
            input, args.scale_factor_h, args.scale_factor_w, args.mode));
}

UpsampleOperation::spec_return_value_t UpsampleOperation::compute_output_specs(
    const operation_attributes_t& args, const Tensor& input) {
    const auto& s = input.logical_shape();
    const uint32_t out_n = s[0];
    const uint32_t out_h = static_cast<uint32_t>(std::floor(s[1] * args.scale_factor_h));
    const uint32_t out_w = static_cast<uint32_t>(std::floor(s[2] * args.scale_factor_w));
    const uint32_t out_c = s[3];
    const ttnn::Shape output_shape({out_n, out_h, out_w, out_c});

    constexpr auto output_layout = tt::tt_metal::Layout::ROW_MAJOR;
    const auto output_dtype =
        input.dtype() == tt::tt_metal::DataType::BFLOAT8_B ? tt::tt_metal::DataType::BFLOAT16 : input.dtype();
    auto make_spec = [&](const tt::tt_metal::MemoryConfig& mc) {
        return tt::tt_metal::TensorSpec(
            output_shape, tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(output_layout), mc));
    };

    if (!args.output_mem_config.is_sharded()) {
        return make_spec(args.output_mem_config);
    }
    TT_FATAL(
        input.memory_config().is_sharded(), "Output memory config is sharded but input memory config is not sharded");
    return make_spec(upsample_common::compute_integer_output_mem_config(
        args.output_mem_config, input, args.mode, args.scale_factor_h, args.scale_factor_w, out_n, out_h, out_w));
}

UpsampleOperation::tensor_return_value_t UpsampleOperation::create_output_tensors(
    const operation_attributes_t& args, const Tensor& input) {
    return create_device_tensor(compute_output_specs(args, input), input.device());
}

ttnn::Tensor upsample(
    const ttnn::Tensor& input_tensor,
    const float scale_factor_h,
    const float scale_factor_w,
    const std::string& mode,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    return ttnn::device_operation::launch<UpsampleOperation>(
        UpsampleParams{
            .scale_factor_h = scale_factor_h,
            .scale_factor_w = scale_factor_w,
            .mode = mode,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        input_tensor);
}

}  // namespace ttnn::prim::qsr
