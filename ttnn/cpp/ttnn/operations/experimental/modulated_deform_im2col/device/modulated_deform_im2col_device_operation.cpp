// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modulated_deform_im2col_device_operation.hpp"

#include <string_view>

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::modulated_deform_im2col {

namespace {

using tt::tt_metal::TensorMemoryLayout;

void check_input_tensor(const Tensor& t, std::string_view name, const Tensor& ref) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "modulated_deform_im2col: {} must be on device", name);
    TT_FATAL(t.device() == ref.device(), "modulated_deform_im2col: {} must be on the same device as input", name);
    TT_FATAL(t.dtype() == DataType::BFLOAT16, "modulated_deform_im2col: {} must be BFLOAT16, got {}", name, t.dtype());
    TT_FATAL(
        t.layout() == Layout::ROW_MAJOR, "modulated_deform_im2col: {} must be ROW_MAJOR, got {}", name, t.layout());
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "modulated_deform_im2col: {} must be INTERLEAVED",
        name);
    TT_FATAL(
        t.logical_shape().rank() == 4, "modulated_deform_im2col: {} must be rank 4, got {}", name, t.logical_shape());
    // The kernels address one page per row and read its first bytes, so a row must not carry padding.
    TT_FATAL(
        t.padded_shape()[-1] == t.logical_shape()[-1],
        "modulated_deform_im2col: {} rows must not be padded, got logical {} padded {}",
        name,
        t.logical_shape(),
        t.padded_shape());
}

}  // namespace

Im2colShapes derive_shapes(
    const ModulatedDeformIm2colOperation::operation_attributes_t& attrs,
    const ModulatedDeformIm2colOperation::tensor_args_t& args) {
    check_input_tensor(args.input, "input", args.input);
    check_input_tensor(args.offset, "offset", args.input);
    check_input_tensor(args.mask, "mask", args.input);

    const auto [k_h, k_w] = attrs.kernel_size;
    TT_FATAL(k_h > 0 && k_w > 0, "modulated_deform_im2col: kernel_size must be positive");
    TT_FATAL(
        attrs.stride[0] > 0 && attrs.stride[1] > 0 && attrs.dilation[0] > 0 && attrs.dilation[1] > 0,
        "modulated_deform_im2col: stride and dilation must be positive");

    Im2colShapes s;
    const auto& in = args.input.logical_shape();
    s.batch = in[0];
    s.in_h = in[1];
    s.in_w = in[2];
    s.channels = in[3];
    // A channel stick then fills whole tile rows, so the reader copies it into
    // tiles without a ragged tail and the output columns tile exactly.
    TT_FATAL(
        s.channels > 0 && s.channels % tt::constants::TILE_WIDTH == 0,
        "modulated_deform_im2col: input channels must be a positive multiple of {}, got {}",
        tt::constants::TILE_WIDTH,
        s.channels);

    const int64_t span_h = static_cast<int64_t>(attrs.dilation[0]) * (k_h - 1) + 1;
    const int64_t span_w = static_cast<int64_t>(attrs.dilation[1]) * (k_w - 1) + 1;
    const int64_t padded_h = static_cast<int64_t>(s.in_h) + 2 * attrs.padding[0];
    const int64_t padded_w = static_cast<int64_t>(s.in_w) + 2 * attrs.padding[1];
    TT_FATAL(
        padded_h >= span_h && padded_w >= span_w, "modulated_deform_im2col: the kernel does not fit the padded input");
    const uint32_t span_limit = kOffsetClampPixels;
    TT_FATAL(
        s.in_h <= span_limit && s.in_w <= span_limit && attrs.padding[0] <= span_limit &&
            attrs.padding[1] <= span_limit && span_h <= span_limit && span_w <= span_limit,
        "modulated_deform_im2col: input size, padding and dilated kernel span must each be at most {}, got input "
        "{}x{}, padding {}x{}, span {}x{}",
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

    const auto& off = args.offset.logical_shape();
    TT_FATAL(
        static_cast<uint32_t>(off[0]) == s.batch && static_cast<uint32_t>(off[1]) == s.out_h &&
            static_cast<uint32_t>(off[2]) == s.out_w && static_cast<uint32_t>(off[3]) == 2 * s.kernel_positions,
        "modulated_deform_im2col: offset must be (B, H_out, W_out, 2*K*K) = ({}, {}, {}, {}), got {}",
        s.batch,
        s.out_h,
        s.out_w,
        2 * s.kernel_positions,
        off);
    const auto& m = args.mask.logical_shape();
    TT_FATAL(
        static_cast<uint32_t>(m[0]) == s.batch && static_cast<uint32_t>(m[1]) == s.out_h &&
            static_cast<uint32_t>(m[2]) == s.out_w && static_cast<uint32_t>(m[3]) == s.kernel_positions,
        "modulated_deform_im2col: mask must be (B, H_out, W_out, K*K) = ({}, {}, {}, {}), got {}",
        s.batch,
        s.out_h,
        s.out_w,
        s.kernel_positions,
        m);
    return s;
}

void ModulatedDeformIm2colOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(
        attrs.output_memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "modulated_deform_im2col: output_memory_config must be INTERLEAVED");
    (void)derive_shapes(attrs, args);
}

void ModulatedDeformIm2colOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

ModulatedDeformIm2colOperation::spec_return_value_t ModulatedDeformIm2colOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto s = derive_shapes(attrs, args);
    // conv2d's (1, 1, B*H*W, C) layout, so the matmul that follows reads it as is.
    Shape out_shape({1, 1, s.rows, s.kernel_positions * s.channels});
    return tt::tt_metal::TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), attrs.output_memory_config));
}

ModulatedDeformIm2colOperation::tensor_return_value_t ModulatedDeformIm2colOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return create_device_tensor(compute_output_specs(attrs, args), args.input.device());
}

}  // namespace ttnn::operations::experimental::modulated_deform_im2col

namespace ttnn::prim {

ttnn::Tensor modulated_deform_im2col(
    const Tensor& input,
    const Tensor& offset,
    const Tensor& mask,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::modulated_deform_im2col::ModulatedDeformIm2colOperation;
    auto attrs = OperationType::operation_attributes_t{
        .output_memory_config = memory_config.value_or(input.memory_config()),
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
        .dilation = dilation,
    };
    auto args = OperationType::tensor_args_t{.input = input, .offset = offset, .mask = mask};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
