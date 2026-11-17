// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.hpp"
#include "ttnn/operations/copy/typecast/device/typecast_device_op.hpp"
#include "ttnn/operations/core/core.hpp"  // for to_dtype
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

}  // namespace

inline Tensor typecast_impl(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    // Handle host tensors by delegating to to_dtype
    if (is_cpu_tensor(input_tensor)) {
        TT_FATAL(
            !optional_output_tensor.has_value(),
            "Preallocated output tensor is not supported for host tensor typecast. "
            "Use to_dtype directly if you need this functionality.");
        TT_FATAL(
            !sub_core_grids.has_value(),
            "sub_core_grids is not supported for host tensor typecast (only applicable to device operations).");
        // For host tensors, memory_config is not applicable, so we ignore it
        return ttnn::to_dtype(input_tensor, output_dtype);
    }

    // Device tensor path
    DataType input_dtype = input_tensor.dtype();
    // Zero the implicit tile padding before packing to a block-float format: the pack reads whole
    // 16-lane blocks, padding included. The fill writes in place, so it must run on a copy to keep
    // the caller's tensor untouched. Block-aligned widths take the direct path unchanged.
    Tensor typecast_input = input_tensor;
    if (needs_zeroed_tile_padding(input_tensor, output_dtype)) {
        typecast_input = ttnn::clone(input_tensor, std::nullopt, std::nullopt, std::nullopt);
        typecast_input = ttnn::fill_implicit_tile_padding(typecast_input, 0.0f);
    }
    bool preserve_fp32_precision =
        (input_dtype == DataType::FLOAT32) or
        (is_8_bit_int(output_dtype) and (input_dtype == DataType::BFLOAT16 or input_dtype == DataType::BFLOAT8_B or
                                         input_dtype == DataType::BFLOAT4_B)) or
        (input_dtype == DataType::UINT16 and is_8_bit_int(output_dtype)) or is_8_bit_int(input_dtype);
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32 or
                            output_dtype == DataType::FP8_E4M3 or input_dtype == DataType::FP8_E4M3;
    bool bfp8_pack_precise = (output_dtype == DataType::BFLOAT8_B);
    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    return ttnn::prim::typecast(
        typecast_input,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor,
        sub_core_grids);
}

}  // namespace ttnn::operations::copy::detail

namespace ttnn {

Tensor typecast(
    const Tensor& input,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    return operations::copy::detail::typecast_impl(
        input, output_dtype, memory_config_arg, optional_output_tensor, sub_core_grids);
}

Tensor typecast(
    const Tensor& input_tensor,
    const DataType& tt_input_dtype,
    const DataType& tt_output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(tt_input_dtype == input_tensor.dtype(), "input dtype and input tensor's dtype provided should match");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            tt_output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }
    return operations::copy::detail::typecast_impl(
        input_tensor, tt_output_dtype, memory_config, optional_output_tensor, sub_core_grids);
}

}  // namespace ttnn
