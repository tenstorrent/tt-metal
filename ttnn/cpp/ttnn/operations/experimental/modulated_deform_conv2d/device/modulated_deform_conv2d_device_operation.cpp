// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modulated_deform_conv2d_device_operation.hpp"

#include <algorithm>
#include <string_view>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/core/program_cache_l1.hpp"
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
    // The compute kernel calls the Blackhole and Wormhole SFPU functions directly.
    TT_FATAL(args.input.device()->arch() != tt::ARCH::QUASAR, "{} is not supported on Quasar", op);
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
    // Whole tiles of channels: compute tilizes each corner block into them, the channels split into
    // whole per-pass chunks, and the output columns tile exactly.
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
        s.in_h + attrs.padding[0] <= span_limit && s.in_w + attrs.padding[1] <= span_limit && span_h <= span_limit &&
            span_w <= span_limit,
        "{}: each input size plus its padding, and the dilated kernel span, must be at most {}, got input "
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

    // The kernels size and step through every TILE operand as default 32x32 tiles in row order.
    const auto default_tile = [](const Tensor& t) {
        const auto& tile = t.tensor_spec().tile();
        return tile == tt::tt_metal::Tile{} && !tile.get_transpose_within_face() && !tile.get_transpose_of_faces();
    };

    const auto& w = args.weight;
    TT_FATAL(w.storage_type() == StorageType::DEVICE, "{}: weight must be on device", op);
    TT_FATAL(w.device() == args.input.device(), "{}: weight must be on the input's device", op);
    TT_FATAL(w.dtype() == DataType::BFLOAT16, "{}: weight must be BFLOAT16, got {}", op, w.dtype());
    TT_FATAL(w.layout() == Layout::TILE, "{}: weight must be TILE, got {}", op, w.layout());
    TT_FATAL(default_tile(w), "{}: weight must use the default non-transposed 32x32 tile", op);
    TT_FATAL(
        w.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "{}: weight must be INTERLEAVED", op);
    const auto& ws = w.logical_shape();
    TT_FATAL(
        ws.rank() == 2 && static_cast<uint32_t>(ws[0]) == s.kernel_positions * s.channels && ws[1] > 0 &&
            ws[1] % tt::constants::TILE_WIDTH == 0,
        "{}: weight must be (K*K*C, C_out) = ({}, C_out), C_out a positive multiple of {}, got {}",
        op,
        s.kernel_positions * s.channels,
        tt::constants::TILE_WIDTH,
        ws);
    s.out_channels = ws[1];
    if (args.bias.has_value()) {
        const auto& b = *args.bias;
        TT_FATAL(b.storage_type() == StorageType::DEVICE, "{}: bias must be on device", op);
        TT_FATAL(b.device() == args.input.device(), "{}: bias must be on the input's device", op);
        TT_FATAL(
            b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "{}: bias must be INTERLEAVED", op);
        TT_FATAL(b.dtype() == DataType::BFLOAT16, "{}: bias must be BFLOAT16, got {}", op, b.dtype());
        TT_FATAL(b.layout() == Layout::TILE, "{}: bias must be TILE, got {}", op, b.layout());
        TT_FATAL(default_tile(b), "{}: bias must use the default non-transposed 32x32 tile", op);
        TT_FATAL(
            b.logical_shape().rank() == 2 && b.logical_shape()[0] == 1 &&
                static_cast<uint32_t>(b.logical_shape()[1]) == s.out_channels,
            "{}: bias must be (1, {}), got {}",
            op,
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

namespace {

// Bytes `spec` takes in its fullest L1 bank. The L1 free lists hand out blocks in DRAM-alignment
// granules, so even a small tensor costs a whole granule.
uint64_t l1_bytes_per_bank(const tt::tt_metal::TensorSpec& spec, const tt::tt_metal::Allocator& allocator) {
    using tt::tt_metal::BufferType;
    const uint64_t granule = allocator.get_alignment(BufferType::DRAM);
    const uint64_t page_bytes = spec.compute_consumed_memory_bytes_per_bank(
        allocator.get_alignment(BufferType::L1), allocator.get_num_banks(BufferType::L1));
    return tt::align(std::max(page_bytes, granule), granule);
}

}  // namespace

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
    using namespace ttnn::operations::experimental::modulated_deform_conv2d;
    using OperationType = ModulatedDeformConv2dOperation;
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
    // The CBs may take the L1 they can reach at this call, short of the output, which is allocated
    // after this and below the current frontier when it goes to L1. Without a device input the
    // budget stays 0 and validation reports it.
    if (const auto* device = input.device(); device != nullptr) {
        uint64_t l1_free = ttnn::operations::core::available_program_l1_capacity(device);
        if (attrs.output_memory_config.buffer_type() == tt::tt_metal::BufferType::L1) {
            const uint64_t output_bytes =
                l1_bytes_per_bank(OperationType::compute_output_specs(attrs, args), *device->allocator());
            l1_free = l1_free > output_bytes ? l1_free - output_bytes : 0;
        }
        attrs.l1_budget = static_cast<uint32_t>(l1_free / kL1BudgetGranule * kL1BudgetGranule);
    }
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
