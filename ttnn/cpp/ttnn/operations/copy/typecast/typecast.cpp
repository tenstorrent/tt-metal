// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"
#include "ttnn/operations/copy/typecast/device/typecast_device_op.hpp"
#include "ttnn/operations/core/core.hpp"  // for to_dtype, to_memory_config
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/tensor/tensor_utils.hpp"  // for is_cpu_tensor

#include <tt-metalium/constants.hpp>

namespace ttnn::operations::copy::detail {

namespace {

// A BFLOAT8_B / BFLOAT4_B pack derives one shared exponent per 16-lane block (one face row) of
// the tile it packs. When the logical width is not a multiple of that block, the implicit width
// padding shares the last 16-lane block with real lanes, so stale padding (an inf left by
// reciprocal/rsqrt/log of zeroed padding, or any large value) collapses those real lanes.
bool needs_zeroed_tile_padding(const Tensor& input_tensor, const DataType& output_dtype) {
    if (input_tensor.layout() != Layout::TILE) {
        return false;
    }
    if (output_dtype != DataType::BFLOAT8_B && output_dtype != DataType::BFLOAT4_B) {
        return false;
    }
    const DataType input_dtype = input_tensor.dtype();
    if (input_dtype != DataType::BFLOAT16 && input_dtype != DataType::FLOAT32) {
        return false;
    }
    return input_tensor.logical_shape()[-1] % tt::constants::FACE_WIDTH != 0;
}

Tensor typecast_on_device(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Layout>& output_layout = std::nullopt) {
    const DataType input_dtype = input_tensor.dtype();
    // Zero the implicit tile padding before packing to a block-float format: the pack reads whole
    // 16-lane blocks, padding included. The fill writes in place, so it must run on a copy to keep
    // the caller's tensor untouched. Block-aligned widths take the direct path unchanged.
    Tensor typecast_input = input_tensor;
    if (needs_zeroed_tile_padding(input_tensor, output_dtype)) {
        typecast_input = ttnn::clone(input_tensor, std::nullopt, std::nullopt, std::nullopt, sub_core_grids);
        typecast_input = ttnn::fill_implicit_tile_padding(typecast_input, 0.0f, std::nullopt, sub_core_grids);
    }
    const bool preserve_fp32_precision =
        (input_dtype == DataType::FLOAT32) or
        (is_8_bit_int(output_dtype) and (input_dtype == DataType::BFLOAT16 or input_dtype == DataType::BFLOAT8_B or
                                         input_dtype == DataType::BFLOAT4_B)) or
        (input_dtype == DataType::UINT16 and is_8_bit_int(output_dtype)) or is_8_bit_int(input_dtype);
    const bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                                  output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                                  input_dtype == DataType::UINT32 or input_dtype == DataType::INT32 or
                                  output_dtype == DataType::FP8_E4M3 or input_dtype == DataType::FP8_E4M3;
    const bool bfp8_pack_precise = (output_dtype == DataType::BFLOAT8_B);
    return ttnn::prim::typecast(
        typecast_input,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor,
        sub_core_grids,
        output_layout);
}

bool row_major_pages_match(
    const Tensor& input_tensor, const DataType& output_dtype, const MemoryConfig& memory_config) {
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        return true;
    }
    const auto output_spec = TensorSpec(
        input_tensor.logical_shape(), tt::tt_metal::TensorLayout(output_dtype, Layout::ROW_MAJOR, memory_config));
    return ttnn::prim::row_major_page_elements_match(input_tensor.tensor_spec(), output_spec);
}

Tensor typecast_device(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (row_major_pages_match(input_tensor, output_dtype, output_memory_config)) {
        return typecast_on_device(
            input_tensor, output_dtype, output_memory_config, optional_output_tensor, sub_core_grids);
    }
    // Width or block sharded row-major pages hold part of a row; move through full-row DRAM pages.
    auto source = input_tensor;
    if (!row_major_pages_match(input_tensor, output_dtype, ttnn::DRAM_MEMORY_CONFIG)) {
        source = ttnn::to_memory_config(input_tensor, ttnn::DRAM_MEMORY_CONFIG);
    }
    if (row_major_pages_match(source, output_dtype, output_memory_config)) {
        return typecast_on_device(source, output_dtype, output_memory_config, optional_output_tensor, sub_core_grids);
    }
    const auto interleaved =
        typecast_on_device(source, output_dtype, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, sub_core_grids);
    return ttnn::to_memory_config(interleaved, output_memory_config, std::nullopt, optional_output_tensor);
}

// Padded device tilize takes no 8-bit integers; they are widened first, which keeps every value.
DataType tilizable_dtype(DataType dtype) {
    if (dtype == DataType::UINT8) {
        return DataType::UINT16;
    }
    if (dtype == DataType::INT8) {
        return DataType::INT32;
    }
    return dtype;
}

// Fused in one pass when possible. Otherwise tilize runs before the cast and untilize after it, so
// block-float data only ever lives in TILE layout.
Tensor typecast_layout_change(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const Layout& output_layout,
    const MemoryConfig& output_memory_config) {
    const auto output_spec = TensorSpec(
        input_tensor.logical_shape(), tt::tt_metal::TensorLayout(output_dtype, output_layout, output_memory_config));
    if (ttnn::prim::can_fuse_typecast_layout_change(input_tensor, output_spec, std::nullopt)) {
        return typecast_on_device(
            input_tensor, output_dtype, output_memory_config, std::nullopt, std::nullopt, output_layout);
    }
    if (output_layout == Layout::TILE) {
        const auto tilize_dtype = tilizable_dtype(input_tensor.dtype());
        const auto tilize_input =
            tilize_dtype == input_tensor.dtype()
                ? input_tensor
                : typecast_device(input_tensor, tilize_dtype, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, std::nullopt);
        const auto tiled = ttnn::to_layout(tilize_input, Layout::TILE, std::nullopt, ttnn::DRAM_MEMORY_CONFIG);
        return typecast_device(tiled, output_dtype, output_memory_config, std::nullopt, std::nullopt);
    }
    const auto cast = typecast_device(input_tensor, output_dtype, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, std::nullopt);
    return ttnn::to_layout(cast, Layout::ROW_MAJOR, std::nullopt, output_memory_config);
}

Tensor typecast_host(const Tensor& input_tensor, const DataType& output_dtype, const Layout& output_layout) {
    if (output_layout == input_tensor.layout()) {
        return ttnn::to_dtype(input_tensor, output_dtype);
    }
    if (output_layout == Layout::TILE) {
        return ttnn::to_dtype(ttnn::to_layout(input_tensor, Layout::TILE), output_dtype);
    }
    return ttnn::to_layout(ttnn::to_dtype(input_tensor, output_dtype), Layout::ROW_MAJOR);
}

}  // namespace

inline Tensor typecast_impl(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Layout>& output_layout_arg = std::nullopt) {
    const Layout output_layout = optional_output_tensor.has_value() ? optional_output_tensor->layout()
                                                                    : output_layout_arg.value_or(input_tensor.layout());
    TT_FATAL(
        !output_layout_arg.has_value() || output_layout_arg.value() == output_layout,
        "If both output layout and output tensor provided layout should match");
    TT_FATAL(
        !(output_layout == Layout::ROW_MAJOR && is_block_float(output_dtype)),
        "Typecast to {} requires TILE output layout",
        output_dtype);

    // Handle host tensors by delegating to to_dtype / to_layout
    if (is_cpu_tensor(input_tensor)) {
        TT_FATAL(
            !optional_output_tensor.has_value(),
            "Preallocated output tensor is not supported for host tensor typecast. "
            "Use to_dtype directly if you need this functionality.");
        TT_FATAL(
            !sub_core_grids.has_value(),
            "sub_core_grids is not supported for host tensor typecast (only applicable to device operations).");
        // For host tensors, memory_config is not applicable, so we ignore it
        return typecast_host(input_tensor, output_dtype, output_layout);
    }

    const auto output_memory_config = optional_output_tensor.has_value()
                                          ? optional_output_tensor.value().memory_config()
                                          : memory_config.value_or(input_tensor.memory_config());
    if (output_layout != input_tensor.layout()) {
        TT_FATAL(
            !optional_output_tensor.has_value(),
            "Typecast with a layout change does not support a preallocated output tensor");
        TT_FATAL(!sub_core_grids.has_value(), "Typecast with a layout change does not support sub_core_grids");
        return typecast_layout_change(input_tensor, output_dtype, output_layout, output_memory_config);
    }
    return typecast_device(input_tensor, output_dtype, output_memory_config, optional_output_tensor, sub_core_grids);
}

}  // namespace ttnn::operations::copy::detail

namespace ttnn {

Tensor typecast(
    const Tensor& input,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Layout>& output_layout) {
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    return operations::copy::detail::typecast_impl(
        input, output_dtype, memory_config_arg, optional_output_tensor, sub_core_grids, output_layout);
}

Tensor typecast(
    const Tensor& input_tensor,
    const DataType& tt_input_dtype,
    const DataType& tt_output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Layout>& output_layout) {
    TT_FATAL(tt_input_dtype == input_tensor.dtype(), "input dtype and input tensor's dtype provided should match");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            tt_output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }
    return operations::copy::detail::typecast_impl(
        input_tensor, tt_output_dtype, memory_config, optional_output_tensor, sub_core_grids, output_layout);
}

}  // namespace ttnn
