// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_dataflow.hpp
// Do not include directly - include reduce_helpers_dataflow.hpp instead

#include "llk_defs.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace dataflow_kernel_lib {

// Row size in uint32 for bfloat16 (8 u32 = 16 bf16)
constexpr uint32_t ROW_SIZE_U32 = 8;

// Row size in uint32 for float32 (16 u32 = 16 f32)
constexpr uint32_t ROW_SIZE_U32_FP32 = 16;

// =============================================================================
// Float to scaler bit conversion
// =============================================================================

template <DataFormat data_format>
FORCE_INLINE uint32_t float_to_scaler_bits(float value) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "float_to_scaler_bits only supports Float16_b (bfloat16) and Float32 formats");

    const uint32_t bits = __builtin_bit_cast(uint32_t, value);

    if constexpr (data_format == DataFormat::Float32) {
        return bits;
    } else {
        // Float16_b (bfloat16): pack two bf16 values into one uint32
        uint16_t bf16 = static_cast<uint16_t>(bits >> 16);
        return (static_cast<uint32_t>(bf16) << 16) | bf16;
    }
}

template <DataFormat data_format>
FORCE_INLINE void fill_face_row0_cols(volatile tt_l1_ptr uint32_t* face_ptr, uint32_t scaler, uint32_t cols_in_face) {
    if constexpr (data_format == DataFormat::Float32) {
        for (uint32_t col = 0; col < cols_in_face; ++col) {
            face_ptr[col] = scaler;
        }
    } else {
        const uint32_t full_pairs = cols_in_face / 2;
        for (uint32_t col = 0; col < full_pairs; ++col) {
            face_ptr[col] = scaler;
        }
        if (cols_in_face & 1) {
            // Lower 16 bits = first column in pair (RISC-V little-endian)
            face_ptr[full_pairs] = scaler & 0x0000FFFFu;
        }
    }
}

template <DataFormat data_format, uint32_t face_rows, uint32_t faces_per_row>
FORCE_INLINE void fill_first_row_valid_columns(
    volatile tt_l1_ptr uint32_t* ptr, uint32_t value, uint32_t valid_columns) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_first_row_valid_columns only supports Float16_b and Float32 formats");

    constexpr uint32_t face_size_u32 = (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        for (uint32_t face_col = 0; face_col < faces_per_row; ++face_col) {
            uint32_t cols_in_face = 0;
            const uint32_t face_col_start = face_col * tt::constants::FACE_WIDTH;
            if (valid_columns > face_col_start) {
                const uint32_t remaining = valid_columns - face_col_start;
                cols_in_face = remaining < tt::constants::FACE_WIDTH ? remaining : tt::constants::FACE_WIDTH;
            }
            if (cols_in_face > 0) {
                const uint32_t face_idx = face_row * faces_per_row + face_col;
                fill_face_row0_cols<data_format>(ptr + face_idx * face_size_u32, value, cols_in_face);
            }
        }
    }
}

template <DataFormat data_format, uint32_t face_rows, uint32_t faces_per_row>
FORCE_INLINE void fill_first_row_per_face_row(volatile tt_l1_ptr uint32_t* ptr, uint32_t value, uint32_t valid_rows) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_first_row_per_face_row only supports Float16_b and Float32 formats");

    constexpr uint32_t face_size_u32 = (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        uint32_t columns_in_face_row = 0;
        const uint32_t face_row_start = face_row * tt::constants::FACE_HEIGHT;
        if (valid_rows > face_row_start) {
            const uint32_t remaining = valid_rows - face_row_start;
            columns_in_face_row = remaining < tt::constants::FACE_HEIGHT ? remaining : tt::constants::FACE_HEIGHT;
        }
        for (uint32_t face_col = 0; face_col < faces_per_row; ++face_col) {
            if (columns_in_face_row > 0) {
                const uint32_t face_idx = face_row * faces_per_row + face_col;
                fill_face_row0_cols<data_format>(ptr + face_idx * face_size_u32, value, columns_in_face_row);
            }
        }
    }
}

// =============================================================================
// Format-aware fill_each_face_col0_partial — fills COLUMN 0 of each left face for the first
// `valid_rows` rows (a col-0 mask, consumed by mul_tiles_bcast_cols for a partial REDUCE_COL). Only the
// left face-column is written; bcast_cols broadcasts col 0 across, so the rest is don't-care (zeroed).
// =============================================================================
template <DataFormat data_format, uint32_t face_rows, uint32_t faces_per_row>
FORCE_INLINE void fill_each_face_col0_partial(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler, uint32_t valid_rows) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_each_face_col0_partial only supports Float16_b (bfloat16) and Float32 formats");

    constexpr uint32_t face_size_u32 = (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t row_size_u32 = (data_format == DataFormat::Float32) ? ROW_SIZE_U32_FP32 : ROW_SIZE_U32;
    constexpr uint32_t rows_per_face = tt::constants::FACE_HEIGHT;

    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        const uint32_t face_row_start = face_row * rows_per_face;
        uint32_t rows_in_face = 0;
        if (valid_rows > face_row_start) {
            const uint32_t remaining = valid_rows - face_row_start;
            rows_in_face = remaining < rows_per_face ? remaining : rows_per_face;
        }
        // left face only (face_col == 0); write column 0 of each valid row (fill_face_row0_cols with a
        // single column lands on col 0, incl. the bf16 low-16-bits case).
        volatile tt_l1_ptr uint32_t* face_ptr = ptr + (face_row * faces_per_row) * face_size_u32;
        for (uint32_t r = 0; r < rows_in_face; ++r) {
            fill_face_row0_cols<data_format>(face_ptr + r * row_size_u32, scaler, 1);
        }
    }
}

namespace reduce_auxiliary_detail {

template <uint32_t cb_id, typename Tile>
FORCE_INLINE void prepare_tile() {
    constexpr auto tile_type = Tile::type;
    constexpr DataFormat data_format = get_dataformat(cb_id);
    constexpr uint32_t tile_r_dim = get_tile_r_dim<cb_id>();
    constexpr uint32_t tile_c_dim = get_tile_c_dim<cb_id>();
    static_assert(tile_r_dim % tt::constants::FACE_HEIGHT == 0, "tile height must be a multiple of FACE_HEIGHT");
    static_assert(tile_c_dim % tt::constants::FACE_WIDTH == 0, "tile width must be a multiple of FACE_WIDTH");
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "reduction auxiliary tiles only support Float16_b and Float32 formats");
    constexpr uint32_t face_rows = tile_r_dim / tt::constants::FACE_HEIGHT;
    constexpr uint32_t faces_per_row = tile_c_dim / tt::constants::FACE_WIDTH;

    if constexpr (tile_type == ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstRow) {
        static_assert(
            Tile::num_valid_elements <= tile_c_dim,
            "FirstRow auxiliary tile valid-element count exceeds the tile width");
    } else if constexpr (
        tile_type == ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstColumn ||
        tile_type == ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstRowPerFaceRow) {
        static_assert(
            Tile::num_valid_elements <= tile_r_dim,
            "Column-oriented auxiliary tile valid-element count exceeds the tile height");
    } else {
        static_assert(
            tile_type == ttnn::kernel_lib::ReduceAuxiliaryTileType::Zero, "Unknown reduction auxiliary tile type");
    }

    DataflowBuffer dfb(cb_id);
    dfb.reserve_back(1);
    const uint32_t write_addr = dfb.get_write_ptr();

    Noc noc;
    noc.async_write_zeros(dfb, get_tile_size(cb_id));
    noc.write_zeros_l1_barrier();

    if constexpr (tile_type != ttnn::kernel_lib::ReduceAuxiliaryTileType::Zero) {
        const float value = __builtin_bit_cast(float, static_cast<uint32_t>(Tile::value_bits));
        const uint32_t packed_value = float_to_scaler_bits<data_format>(value);
        if (packed_value != 0) {
            if constexpr (tile_type == ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstRow) {
                fill_first_row_valid_columns<data_format, face_rows, faces_per_row>(
                    addr_to_l1_ptr(write_addr), packed_value, Tile::num_valid_elements);
            } else if constexpr (tile_type == ttnn::kernel_lib::ReduceAuxiliaryTileType::FirstColumn) {
                fill_each_face_col0_partial<data_format, face_rows, faces_per_row>(
                    addr_to_l1_ptr(write_addr), packed_value, Tile::num_valid_elements);
            } else {
                fill_first_row_per_face_row<data_format, face_rows, faces_per_row>(
                    addr_to_l1_ptr(write_addr), packed_value, Tile::num_valid_elements);
            }
        }
    }

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    flush_l2_cache_range(write_addr, get_tile_size(cb_id));
#endif
    dfb.push_back(1);
}

template <typename Auxiliary, uint32_t tile_index = 0>
FORCE_INLINE void prepare_tiles() {
    if constexpr (tile_index < Auxiliary::num_tiles) {
        prepare_tile<Auxiliary::cb_id, typename Auxiliary::template Tile<tile_index>>();
        prepare_tiles<Auxiliary, tile_index + 1>();
    }
}

}  // namespace reduce_auxiliary_detail

template <typename Auxiliary>
FORCE_INLINE void prepare_reduce_auxiliary_tiles() {
    reduce_auxiliary_detail::prepare_tiles<Auxiliary>();
}

}  // namespace dataflow_kernel_lib
