// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modulated_deform_conv2d_device_operation.hpp"

#include <string_view>

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::modulated_deform_conv2d {

namespace {

using tt::tt_metal::TensorMemoryLayout;

constexpr std::string_view op = "modulated_deform_conv2d";

void check_input_tensor(const Tensor& t, std::string_view name, const Tensor& ref, bool may_be_sharded = false) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{}: {} must be on device", op, name);
    TT_FATAL(t.device() == ref.device(), "{}: {} must be on the same device as input", op, name);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "{}: {} must be BFLOAT16, got {}", op, name, t.dtype());
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "{}: {} must be ROW_MAJOR, got {}", op, name, t.layout());
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
            (may_be_sharded && t.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
        "{}: {} must be INTERLEAVED{}, got {}",
        op,
        name,
        may_be_sharded ? " or HEIGHT_SHARDED" : "",
        t.memory_config().memory_layout());
    TT_FATAL(t.logical_shape().rank() == 4, "{}: {} must be rank 4, got {}", op, name, t.logical_shape());
    // The kernels address one page per row, its channels from the first byte, so a row must not
    // carry padding.
    TT_FATAL(
        t.padded_shape()[-1] == t.logical_shape()[-1],
        "{}: {} rows must not be padded, got logical {} padded {}",
        op,
        name,
        t.logical_shape(),
        t.padded_shape());
}

}  // namespace

DeformConvShapes derive_shapes(
    const ModulatedDeformConv2dOperation::operation_attributes_t& attrs,
    const ModulatedDeformConv2dOperation::tensor_args_t& args) {
    check_input_tensor(args.input, "input", args.input, /*may_be_sharded=*/true);
    check_input_tensor(args.offset_mask, "offset_mask", args.input);

    const auto [k_h, k_w] = attrs.kernel_size;
    TT_FATAL(k_h > 0 && k_w > 0, "{}: kernel_size must be positive", op);
    // The geometry tiles hold one kernel position per column.
    TT_FATAL(k_h * k_w <= 32, "{}: kernel_size {}x{} has more than 32 positions", op, k_h, k_w);
    TT_FATAL(
        attrs.stride[0] > 0 && attrs.stride[1] > 0 && attrs.dilation[0] > 0 && attrs.dilation[1] > 0,
        "{}: stride and dilation must be positive",
        op);

    DeformConvShapes s;
    const auto& in = args.input.logical_shape();
    s.batch = in[0];
    s.in_h = in[1];
    s.in_w = in[2];
    s.channels = in[3];
    // A channel stick then fills whole tile rows, so the reader copies it into
    // tiles without a ragged tail and the output columns tile exactly.
    TT_FATAL(
        s.channels > 0 && s.channels % tt::constants::TILE_WIDTH == 0,
        "{}: input channels must be a positive multiple of {}, got {}",
        op,
        tt::constants::TILE_WIDTH,
        s.channels);

    const int64_t span_h = static_cast<int64_t>(attrs.dilation[0]) * (k_h - 1) + 1;
    const int64_t span_w = static_cast<int64_t>(attrs.dilation[1]) * (k_w - 1) + 1;
    const int64_t padded_h = static_cast<int64_t>(s.in_h) + 2 * attrs.padding[0];
    const int64_t padded_w = static_cast<int64_t>(s.in_w) + 2 * attrs.padding[1];
    TT_FATAL(padded_h >= span_h && padded_w >= span_w, "{}: the kernel does not fit the padded input", op);
    const uint32_t span_limit = kOffsetClampPixels;
    TT_FATAL(
        s.in_h <= span_limit && s.in_w <= span_limit && attrs.padding[0] <= span_limit &&
            attrs.padding[1] <= span_limit && span_h <= span_limit && span_w <= span_limit,
        "{}: input size, padding and dilated kernel span must each be at most {}, got input "
        "{}x{}, padding {}x{}, span {}x{}",
        op,
        span_limit,
        s.in_h,
        s.in_w,
        attrs.padding[0],
        attrs.padding[1],
        span_h,
        span_w);
    s.out_h = static_cast<uint32_t>((padded_h - span_h) / attrs.stride[0] + 1);
    s.out_w = static_cast<uint32_t>((padded_w - span_w) / attrs.stride[1] + 1);
    s.kernel_positions = k_h * k_w;
    s.rows = s.batch * s.out_h * s.out_w;

    const auto& off = args.offset_mask.logical_shape();
    TT_FATAL(
        static_cast<uint32_t>(off[0]) == s.batch && static_cast<uint32_t>(off[1]) == s.out_h &&
            static_cast<uint32_t>(off[2]) == s.out_w && static_cast<uint32_t>(off[3]) == 3 * s.kernel_positions,
        "{}: offset_mask must be (B, H_out, W_out, 3*K*K) = ({}, {}, {}, {}), got {}",
        op,
        s.batch,
        s.out_h,
        s.out_w,
        3 * s.kernel_positions,
        off);

    // A height-sharded input sets the work split: each core takes the output rows of its own shard,
    // so a shard must hold whole blocks of 32 rows and the output rows must be the input's.
    if (args.input.memory_config().is_sharded()) {
        const auto& spec = *args.input.shard_spec();
        TT_FATAL(
            args.input.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
            "{}: a sharded input must be in L1",
            op);
        TT_FATAL(
            spec.shape[1] == s.channels,
            "{}: input shards must hold whole rows of {} channels, got {}",
            op,
            s.channels,
            spec.shape[1]);
        TT_FATAL(
            spec.shape[0] % tt::constants::TILE_HEIGHT == 0,
            "{}: input shard height must be a multiple of {}, got {}",
            op,
            tt::constants::TILE_HEIGHT,
            spec.shape[0]);
        TT_FATAL(
            s.out_h == s.in_h && s.out_w == s.in_w,
            "{}: a sharded input needs the output's height and width, got input {}x{}, output {}x{}",
            op,
            s.in_h,
            s.in_w,
            s.out_h,
            s.out_w);
    }

    const auto& w = args.weight;
    TT_FATAL(w.storage_type() == StorageType::DEVICE, "modulated_deform_conv2d: weight must be on device");
    TT_FATAL(w.device() == args.input.device(), "modulated_deform_conv2d: weight must be on the input's device");
    TT_FATAL(w.dtype() == DataType::BFLOAT16, "modulated_deform_conv2d: weight must be BFLOAT16, got {}", w.dtype());
    TT_FATAL(w.layout() == Layout::TILE, "modulated_deform_conv2d: weight must be TILE, got {}", w.layout());
    TT_FATAL(
        w.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "modulated_deform_conv2d: weight must be INTERLEAVED");
    const auto& ws = w.logical_shape();
    TT_FATAL(
        ws.rank() == 2 && static_cast<uint32_t>(ws[0]) == s.kernel_positions * s.channels && ws[1] > 0 &&
            ws[1] % tt::constants::TILE_WIDTH == 0,
        "modulated_deform_conv2d: weight must be (K*K*C, C_out) = ({}, C_out), C_out a positive multiple of {}, "
        "got {}",
        s.kernel_positions * s.channels,
        tt::constants::TILE_WIDTH,
        ws);
    s.out_channels = ws[1];
    if (args.bias.has_value()) {
        const auto& b = *args.bias;
        TT_FATAL(b.storage_type() == StorageType::DEVICE, "modulated_deform_conv2d: bias must be on device");
        TT_FATAL(b.device() == args.input.device(), "modulated_deform_conv2d: bias must be on the input's device");
        TT_FATAL(
            b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "modulated_deform_conv2d: bias must be INTERLEAVED");
        TT_FATAL(b.dtype() == DataType::BFLOAT16, "modulated_deform_conv2d: bias must be BFLOAT16, got {}", b.dtype());
        TT_FATAL(b.layout() == Layout::TILE, "modulated_deform_conv2d: bias must be TILE, got {}", b.layout());
        TT_FATAL(
            b.logical_shape().rank() == 2 && b.logical_shape()[0] == 1 &&
                static_cast<uint32_t>(b.logical_shape()[1]) == s.out_channels,
            "modulated_deform_conv2d: bias must be (1, {}), got {}",
            s.out_channels,
            b.logical_shape());
    }
    return s;
}

void ModulatedDeformConv2dOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(
        attrs.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{}: output_memory_config must be INTERLEAVED",
        op);
    (void)derive_shapes(attrs, args);
}

void ModulatedDeformConv2dOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

ModulatedDeformConv2dOperation::spec_return_value_t ModulatedDeformConv2dOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto s = derive_shapes(attrs, args);
    // conv2d's (1, 1, B*H*W, C) layout.
    Shape out_shape({1, 1, s.rows, s.out_channels});
    return tt::tt_metal::TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), attrs.output_memory_config));
}

ModulatedDeformConv2dOperation::tensor_return_value_t ModulatedDeformConv2dOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.input.device());
}

}  // namespace ttnn::operations::experimental::modulated_deform_conv2d

namespace ttnn::prim {

ttnn::Tensor modulated_deform_conv2d(
    const Tensor& input,
    const Tensor& offset_mask,
    const Tensor& weight,
    const std::optional<const Tensor>& bias,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    bool relu,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::modulated_deform_conv2d::ModulatedDeformConv2dOperation;
    auto attrs = OperationType::operation_attributes_t{
        // The output is interleaved; a sharded input's would not be, so its default is DRAM.
        .output_memory_config = memory_config.value_or(
            input.memory_config().is_sharded() ? ttnn::DRAM_MEMORY_CONFIG : input.memory_config()),
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
        .dilation = dilation,
        .relu = relu,
    };
    auto args =
        OperationType::tensor_args_t{.input = input, .offset_mask = offset_mask, .weight = weight, .bias = bias};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
