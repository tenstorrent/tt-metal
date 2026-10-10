// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rgb_to_yuv_device_op.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <cmath>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// Sharded output is not supported and must never silently run. This gate is
// called from compute_output_specs (which the framework runs before validate,
// via create_output_tensors) so it fires before any sharded TensorSpec is
// constructed, and from validate for defense-in-depth on the cache-hit path.
static void reject_unsupported_output_memory_config(const MemoryConfig& mem_cfg) {
    TT_FATAL(
        mem_cfg.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Sharded output is not supported; output memory must be interleaved, got {}",
        mem_cfg.memory_layout());
}

RgbToYuvDeviceOperation::program_factory_t RgbToYuvDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RgbToYuvProgramFactory{};
}

void RgbToYuvDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.input;
    TT_FATAL(in.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(in.buffer() != nullptr, "Input buffer must be allocated");
    TT_FATAL(in.dtype() == DataType::BFLOAT16, "Input must be bfloat16");
    TT_FATAL(in.layout() == Layout::ROW_MAJOR, "Input must be row-major (CHWT)");

    const auto& shape = in.logical_shape();
    const uint32_t p = attrs.input_patch_size;
    if (p == 0) {
        TT_FATAL(shape.rank() == 4, "Input must be 4D (C, H, W, T)");
        TT_FATAL(shape[0] == 3, "Input must have C=3 (RGB)");
        TT_FATAL(
            shape[1] >= 2 && shape[1] % 2 == 0, "H must be even and >= 2 for 4:2:0 subsampling (got {})", shape[1]);
        TT_FATAL(
            shape[2] >= 2 && shape[2] % 2 == 0, "W must be even and >= 2 for 4:2:0 subsampling (got {})", shape[2]);
        TT_FATAL(shape[3] > 0, "T must be positive (got {})", shape[3]);
    } else {
        // An even p keeps each 2-row group inside one patch row, which the reader relies on.
        TT_FATAL(p >= 2 && p % 2 == 0, "input_patch_size must be 0 or an even number >= 2 (got {})", p);
        TT_FATAL(shape.rank() == 5, "Patchified input must be 5D (1, T, H/p, W/p, 3*p*p), got rank {}", shape.rank());
        TT_FATAL(shape[0] == 1, "Patchified input batch must be 1 (got {})", shape[0]);
        TT_FATAL(shape[4] == 3 * p * p, "Patchified input must have 3*p*p = {} channels (got {})", 3 * p * p, shape[4]);
        TT_FATAL(shape[1] > 0 && shape[2] > 0 && shape[3] > 0, "Patchified input dims must be positive ({})", shape);
    }

    // Patchified input may pad its channel dim (conv3d rounds C_out up to a tile): the reader reads
    // each page at the aligned page stride and uses only the first 3*p*p channels.
    const auto& padded = in.padded_shape();
    bool padding_ok = in.logical_shape() == padded;
    if (p != 0 && !padding_ok) {
        padding_ok = padded[4] >= shape[4];
        for (int i = 0; i < 4; i++) {
            padding_ok = padding_ok && padded[i] == shape[i];
        }
    }
    TT_FATAL(padding_ok, "Padded input is not supported (logical {} vs padded {})", in.logical_shape(), padded);

    TT_FATAL(
        in.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input memory must be interleaved, got {}",
        in.memory_config().memory_layout());

    reject_unsupported_output_memory_config(attrs.output_memory_config);

    TT_FATAL(
        attrs.format == YUVFormat::YUV420Planar, "Only YUV 4:2:0 planar output (YUVFormat::YUV420Planar) is supported");

    const auto& c = attrs.coefficients;
    for (int i = 0; i < 4; i++) {
        TT_FATAL(std::isfinite(c.y[i]), "y coefficient [{}] is not finite", i);
        TT_FATAL(std::isfinite(c.cb[i]), "cb coefficient [{}] is not finite", i);
        TT_FATAL(std::isfinite(c.cr[i]), "cr coefficient [{}] is not finite", i);
    }
}

void RgbToYuvDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

RgbToYuvDeviceOperation::spec_return_value_t RgbToYuvDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    reject_unsupported_output_memory_config(attrs.output_memory_config);

    const auto& shape = tensor_args.input.logical_shape();
    // The framework runs create_output_tensors (hence this) before validate, so
    // the rank must be checked here before indexing the shape below.
    const uint32_t p = attrs.input_patch_size;
    uint32_t H, W, T;
    if (p == 0) {
        TT_FATAL(shape.rank() == 4, "Input must be 4D (C, H, W, T), got rank {}", shape.rank());
        H = shape[1], W = shape[2], T = shape[3];
    } else {
        TT_FATAL(shape.rank() == 5, "Patchified input must be 5D (1, T, H/p, W/p, 3*p*p), got rank {}", shape.rank());
        H = shape[2] * p, W = shape[3] * p, T = shape[1];
    }

    auto mem_cfg = attrs.output_memory_config;
    auto uint8_layout = TensorLayout(DataType::UINT8, Layout::ROW_MAJOR, mem_cfg);

    TensorSpec y_spec(ttnn::Shape{1, H, W, T}, uint8_layout);
    TensorSpec u_spec(ttnn::Shape{1, H / 2, W / 2, T}, uint8_layout);
    TensorSpec v_spec(ttnn::Shape{1, H / 2, W / 2, T}, uint8_layout);

    return {y_spec, u_spec, v_spec};
}

RgbToYuvDeviceOperation::tensor_return_value_t RgbToYuvDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [y_spec, u_spec, v_spec] = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input.device();
    return {
        create_device_tensor(y_spec, device),
        create_device_tensor(u_spec, device),
        create_device_tensor(v_spec, device),
    };
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> rgb_to_yuv(
    const Tensor& input,
    const ttnn::experimental::prim::YUVCoefficients& coefficients,
    ttnn::experimental::prim::YUVFormat format,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    uint32_t input_patch_size) {
    using Op = ttnn::experimental::prim::RgbToYuvDeviceOperation;

    auto op_attrs = Op::operation_attributes_t{
        .coefficients = coefficients,
        .format = format,
        .output_memory_config = memory_config.value_or(input.memory_config()),
        .input_patch_size = input_patch_size,
    };
    auto tensor_args = Op::tensor_args_t{.input = input};
    return ttnn::device_operation::launch<Op>(op_attrs, tensor_args);
}

}  // namespace ttnn::prim
