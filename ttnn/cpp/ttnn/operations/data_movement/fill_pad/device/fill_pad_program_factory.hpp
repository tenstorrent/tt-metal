// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fill_pad_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <array>
#include <bit>

namespace ttnn::prim::detail {

inline const std::map<ttnn::DataType, uint32_t> data_type_to_size = {
    {ttnn::DataType::BFLOAT16, 2},
    {ttnn::DataType::FLOAT32, 4},
    {ttnn::DataType::UINT16, 2},
    {ttnn::DataType::UINT32, 4},
    {ttnn::DataType::INT32, 4},
};

// Packs `fill_value` into the uint32 bit pattern expected by the reader/compute kernels.
// Width is deduced from the parameter type: 4-byte types take their raw bit pattern,
// 2-byte integer types are duplicated across both halves of the 32-bit word.
template <typename T>
inline uint32_t pack_fill_value(T fill_value) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "pack_fill_value: unsupported size");
    if constexpr (sizeof(T) == 4) {
        return std::bit_cast<uint32_t>(fill_value);
    } else {
        using U16 = std::uint16_t;
        const uint32_t v = static_cast<uint32_t>(std::bit_cast<U16>(fill_value));
        return (v << 16) | v;
    }
}

// Dtype-directed wrapper used by both program factories. Chooses the native
// fill type per dtype and packs it.
// Float DataTypes (FLOAT32, BFLOAT16) keep the full float32 bit pattern — the
// compute kernel reconstructs it via fill_tile_bitcast and the downstream
// packer handles the bf16 narrowing.
inline uint32_t pack_fill_value_for_dtype(ttnn::DataType dtype, const tt::tt_metal::PadValue& pad_value) {
    // PadValue's uint32_t arm carries an integer value / raw 32-bit pattern (e.g. reduce's int32 pad
    // sentinels, which are not float-representable); the float arm carries a numeric float value (the
    // default for prod, reshape, slice, ...). Mirrors tilize_with_val_padding's get_packed_value.
    if (std::holds_alternative<uint32_t>(pad_value)) {
        const uint32_t fill_value = std::get<uint32_t>(pad_value);
        switch (dtype) {
            // 32-bit integers: the value is already the native bit pattern.
            case ttnn::DataType::INT32:
            case ttnn::DataType::UINT32: return fill_value;
            case ttnn::DataType::UINT16: return pack_fill_value(static_cast<uint16_t>(fill_value));
            case ttnn::DataType::BFLOAT16:
            case ttnn::DataType::FLOAT32: return pack_fill_value(static_cast<float>(fill_value));
            default: TT_THROW("fill_pad: unsupported dtype"); return 0u;
        }
    }
    const float fill_value = std::get<float>(pad_value);
    switch (dtype) {
        case ttnn::DataType::FLOAT32:
        case ttnn::DataType::BFLOAT16: return pack_fill_value(fill_value);
        case ttnn::DataType::UINT16: return pack_fill_value(static_cast<uint16_t>(fill_value));
        case ttnn::DataType::UINT32: return pack_fill_value(static_cast<uint32_t>(fill_value));
        case ttnn::DataType::INT32: return pack_fill_value(static_cast<int32_t>(fill_value));
        default: TT_THROW("fill_pad: unsupported dtype"); return 0u;
    }
}

// DataFormat string for where_tile<> template parameter.
// 2-byte types (BFLOAT16, UINT16) both collapse to Float16_b because that's the only
// 2-byte where_tile<> SFPU variant exposed by the LLKs (matches the ternary_op pattern).
inline std::string get_where_data_fmt(ttnn::DataType dtype) {
    switch (dtype) {
        case ttnn::DataType::FLOAT32: return "DataFormat::Float32";
        case ttnn::DataType::UINT32: return "DataFormat::UInt32";
        case ttnn::DataType::INT32: return "DataFormat::Int32";
        default: return "DataFormat::Float16_b";  // BFLOAT16, UINT16
    }
}

}  // namespace ttnn::prim::detail

namespace ttnn::prim {

// Handles DRAM tensors (Interleaved + DRAM sharded).
//
// Work splitting is unified across three border-tile phases (right / bottom /
// corner). All border tiles across all slices are enumerated into a single
// global index space and handed to split_work_to_cores at tile granularity,
// so 2D / small-N_slices shapes (e.g. 4097x4097) still fan out across the
// full compute grid. Each core gets per-phase (start, num) pairs; phases
// with num == 0 are skipped. A single compute kernel binary covers all cores
// (CT has_right_pad / has_bottom_pad gate the phase branches at compile time).
struct FillPadProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& tensor_return_value);
};

// Handles all L1-sharded tensors (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED).
// Unlike `FillPadProgramFactory` which gives each core a comparable number of borders,
// with FillPadL1ShardedProgramFactory, each core only processes its own local L1 data (no balancing).
struct FillPadL1ShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
