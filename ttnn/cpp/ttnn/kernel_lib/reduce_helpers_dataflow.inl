// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_dataflow.hpp
// Do not include directly - include reduce_helpers_dataflow.hpp instead

#include <cmath>
#include "llk_defs.h"

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

    union {
        float f;
        uint32_t bits;
    } f32_to_bits;
    f32_to_bits.f = value;

    if constexpr (data_format == DataFormat::Float32) {
        return f32_to_bits.bits;
    } else {
        // Float16_b (bfloat16): pack two bf16 values into one uint32
        uint16_t bf16 = static_cast<uint16_t>(f32_to_bits.bits >> 16);
        return (static_cast<uint32_t>(bf16) << 16) | bf16;
    }
}

// =============================================================================
// Float to col-0 scaler bit conversion (matmul-based reduce)
// =============================================================================

template <DataFormat data_format>
FORCE_INLINE uint32_t float_to_col0_scaler_bits(float value) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "float_to_col0_scaler_bits only supports Float16_b (bfloat16) and Float32 formats");

    union {
        float f;
        uint32_t bits;
    } f32_to_bits;
    f32_to_bits.f = value;

    if constexpr (data_format == DataFormat::Float32) {
        return f32_to_bits.bits;
    } else {
        // Float16_b (bfloat16): return lower 16 bits only (col 0 in the u32 pair)
        return static_cast<uint32_t>(static_cast<uint16_t>(f32_to_bits.bits >> 16));
    }
}

// =============================================================================
// Format-aware fill_row0
// =============================================================================

template <DataFormat data_format, bool half_tile>
FORCE_INLINE void fill_row0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_row0 only supports Float16_b (bfloat16) and Float32 formats");

    constexpr uint32_t num_faces = half_tile ? 2 : 4;
    constexpr uint32_t face_size_u32 =
        (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t row_size_u32 =
        (data_format == DataFormat::Float32) ? ROW_SIZE_U32_FP32 : ROW_SIZE_U32;

    for (uint32_t face = 0; face < num_faces; ++face) {
        uint32_t face_offset = face * face_size_u32;
        for (uint32_t column = 0; column < row_size_u32; ++column) {
            ptr[face_offset + column] = scaler;
        }
    }
}

// =============================================================================
// Format-aware fill_row0_partial — fills only first valid_reduce_dim_elements_in_tile columns of row 0
// =============================================================================

template <DataFormat data_format, bool half_tile, uint32_t valid_reduce_dim_elements_in_tile>
FORCE_INLINE void fill_row0_partial(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_row0_partial only supports Float16_b (bfloat16) and Float32 formats");
    static_assert(
        valid_reduce_dim_elements_in_tile > 0 && valid_reduce_dim_elements_in_tile < tt::constants::TILE_WIDTH,
        "valid_reduce_dim_elements_in_tile must be in range [1, TILE_WIDTH-1]");

    constexpr uint32_t num_faces = half_tile ? 2 : 4;
    constexpr uint32_t face_size_u32 =
        (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t COLS_PER_FACE = 16;
    constexpr uint32_t left_cols = valid_reduce_dim_elements_in_tile < COLS_PER_FACE ? valid_reduce_dim_elements_in_tile : COLS_PER_FACE;
    constexpr uint32_t right_cols = valid_reduce_dim_elements_in_tile > COLS_PER_FACE ? valid_reduce_dim_elements_in_tile - COLS_PER_FACE : 0;

    for (uint32_t face = 0; face < num_faces; ++face) {
        uint32_t face_offset = face * face_size_u32;
        uint32_t cols = (face & 1) ? right_cols : left_cols;

        if constexpr (data_format == DataFormat::Float32) {
            for (uint32_t col = 0; col < cols; ++col) {
                ptr[face_offset + col] = scaler;
            }
        } else {
            uint32_t full_pairs = cols / 2;
            for (uint32_t col = 0; col < full_pairs; ++col) {
                ptr[face_offset + col] = scaler;
            }
            if (cols & 1) {
                // Lower 16 bits = first column in pair (RISC-V little-endian)
                ptr[face_offset + full_pairs] = scaler & 0x0000FFFFu;
            }
        }
    }
}

// =============================================================================
// Format-aware fill_col0 (matmul-based reduce — column 0 of left-side faces)
// =============================================================================

template <DataFormat data_format, bool half_tile>
FORCE_INLINE void fill_col0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_col0 only supports Float16_b (bfloat16) and Float32 formats");

    constexpr uint32_t face_size_u32 =
        (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t row_size_u32 =
        (data_format == DataFormat::Float32) ? ROW_SIZE_U32_FP32 : ROW_SIZE_U32;
    constexpr uint32_t ROWS_PER_FACE = 16;

    // Face 0 (top-left)
    for (uint32_t row = 0; row < ROWS_PER_FACE; ++row) {
        ptr[row * row_size_u32] = scaler;
    }
    if constexpr (!half_tile) {
        // Face 2 (bottom-left)
        constexpr uint32_t face2_offset = 2 * face_size_u32;
        for (uint32_t row = 0; row < ROWS_PER_FACE; ++row) {
            ptr[face2_offset + row * row_size_u32] = scaler;
        }
    }
}

// =============================================================================
// Format-aware fill_col0_partial — fills only first valid rows of column 0
// =============================================================================

template <DataFormat data_format, bool half_tile, uint32_t valid_reduce_dim_elements_in_tile>
FORCE_INLINE void fill_col0_partial(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_col0_partial only supports Float16_b (bfloat16) and Float32 formats");
    static_assert(
        valid_reduce_dim_elements_in_tile > 0 && valid_reduce_dim_elements_in_tile < tt::constants::TILE_WIDTH,
        "valid_reduce_dim_elements_in_tile must be in range [1, TILE_WIDTH-1]");

    constexpr uint32_t face_size_u32 =
        (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t row_size_u32 =
        (data_format == DataFormat::Float32) ? ROW_SIZE_U32_FP32 : ROW_SIZE_U32;
    constexpr uint32_t ROWS_PER_FACE = 16;
    constexpr uint32_t top_rows =
        valid_reduce_dim_elements_in_tile < ROWS_PER_FACE ? valid_reduce_dim_elements_in_tile : ROWS_PER_FACE;
    constexpr uint32_t bottom_rows =
        valid_reduce_dim_elements_in_tile > ROWS_PER_FACE ? valid_reduce_dim_elements_in_tile - ROWS_PER_FACE : 0;

    // Face 0 (top-left): first top_rows rows
    for (uint32_t row = 0; row < top_rows; ++row) {
        ptr[row * row_size_u32] = scaler;
    }
    if constexpr (!half_tile && bottom_rows > 0) {
        // Face 2 (bottom-left): first bottom_rows rows
        constexpr uint32_t face2_offset = 2 * face_size_u32;
        for (uint32_t row = 0; row < bottom_rows; ++row) {
            ptr[face2_offset + row * row_size_u32] = scaler;
        }
    }
}

// =============================================================================
// Prepare CB tile for reduce using a caller-provided float scaler
// =============================================================================

template <uint32_t cb_id, PoolType pool_type, ReduceDim reduce_dim,
          uint32_t valid_reduce_dim_elements_in_tile, bool force_reduce_llk>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    static_assert(
        valid_reduce_dim_elements_in_tile > 0 && valid_reduce_dim_elements_in_tile <= tt::constants::TILE_WIDTH,
        "valid_reduce_dim_elements_in_tile must be in range [1, TILE_WIDTH]");

    constexpr DataFormat data_format = get_dataformat(cb_id);
    constexpr bool half_tile = get_tile_num_faces(cb_id) == 2;

    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "prepare_reduce_scaler only supports Float16_b (bfloat16) and Float32 formats");

    // Matmul-based reduce uses col-0 fill; reduce LLK uses row-0 fill
    constexpr bool use_matmul = !force_reduce_llk
                                && (pool_type == PoolType::SUM || pool_type == PoolType::AVG)
                                && reduce_dim == ReduceDim::REDUCE_ROW;

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<data_format, half_tile>(write_addr);

    if constexpr (use_matmul) {
        uint32_t scaler = float_to_col0_scaler_bits<data_format>(scaler_f);
        if (scaler != 0) {
            if constexpr (valid_reduce_dim_elements_in_tile >= tt::constants::TILE_WIDTH) {
                fill_col0<data_format, half_tile>(addr_to_l1_ptr(write_addr), scaler);
            } else {
                fill_col0_partial<data_format, half_tile, valid_reduce_dim_elements_in_tile>(
                    addr_to_l1_ptr(write_addr), scaler);
            }
        }
    } else {
        uint32_t scaler = float_to_scaler_bits<data_format>(scaler_f);
        if (scaler != 0) {
            if constexpr (valid_reduce_dim_elements_in_tile == tt::constants::TILE_WIDTH) {
                fill_row0<data_format, half_tile>(addr_to_l1_ptr(write_addr), scaler);
            } else {
                fill_row0_partial<data_format, half_tile, valid_reduce_dim_elements_in_tile>(
                    addr_to_l1_ptr(write_addr), scaler);
            }
        }
    }

    cb_push_back(cb_id, 1);
}

// =============================================================================
// Format-aware calculate_and_prepare_reduce_scaler (cb_id-deduced format and tile shape)
// =============================================================================

template <
    uint32_t cb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t valid_reduce_dim_elements_in_tile,
    uint32_t reduce_factor,
    bool force_reduce_llk>
FORCE_INLINE void calculate_and_prepare_reduce_scaler() {

    // -------------------------------------------------------------------------
    // 1. Compute scaler value
    //
    //    REDUCE_SCALAR applies scaler twice in LLK (row then col), so use 1/sqrt(N)
    //    REDUCE_ROW/REDUCE_COL apply scaler once, so use 1/N
    //
    //    NOTE: sqrtf() with a runtime argument will link in the software sqrt
    //    implementation. If device memory is tight (~2KB limit), consider
    //    precomputing the scaler on the host instead.
    // -------------------------------------------------------------------------
    float scaler_f;
    if constexpr (pool_type == PoolType::AVG) {
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
            static_assert(reduce_factor > 0, "reduce_factor must be greater than 0");
            scaler_f = 1.0f / sqrtf(static_cast<float>(reduce_factor));
        } else {
            scaler_f = 1.0f / static_cast<float>(reduce_factor);
        }
    } else {
        scaler_f = 1.0f;
    }

    // -------------------------------------------------------------------------
    // 2. Fill the CB with the computed scaler
    // -------------------------------------------------------------------------
    prepare_reduce_scaler<cb_id, pool_type, reduce_dim, valid_reduce_dim_elements_in_tile, force_reduce_llk>(scaler_f);
}

}  // namespace dataflow_kernel_lib
