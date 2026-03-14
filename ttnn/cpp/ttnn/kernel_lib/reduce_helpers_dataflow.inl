// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_dataflow.hpp
// Do not include directly - include reduce_helpers_dataflow.hpp instead

#include <cmath>
#include "llk_defs.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_common.hpp"

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
// Format-aware fill_each_face_row0
// =============================================================================

template <DataFormat data_format, uint32_t num_faces>
FORCE_INLINE void fill_each_face_row0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_each_face_row0 only supports Float16_b (bfloat16) and Float32 formats");

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

template <DataFormat data_format>
FORCE_INLINE void fill_face_col0_rows(volatile tt_l1_ptr uint32_t* face_ptr, uint32_t scaler, uint32_t rows_in_face) {
    constexpr uint32_t row_size_u32 =
        (data_format == DataFormat::Float32) ? ROW_SIZE_U32_FP32 : ROW_SIZE_U32;

    for (uint32_t row = 0; row < rows_in_face; ++row) {
        face_ptr[row * row_size_u32] = scaler;
    }
}

// =============================================================================
// Format-aware fill_each_face_row0_partial — fills row 0 in each participating face
// =============================================================================

template <
    DataFormat data_format,
    ReduceDim reduce_dim,
    uint32_t face_rows,
    uint32_t faces_per_row>
FORCE_INLINE void fill_each_face_row0_partial(
    volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler, uint32_t valid_reduce_dim_elements_in_tile) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_each_face_row0_partial only supports Float16_b (bfloat16) and Float32 formats");
    static_assert(
        reduce_dim == ReduceDim::REDUCE_ROW || reduce_dim == ReduceDim::REDUCE_COL,
        "fill_each_face_row0_partial only supports partial valid elements for REDUCE_ROW and REDUCE_COL");

    constexpr uint32_t face_size_u32 =
        (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;

    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        for (uint32_t face_col = 0; face_col < faces_per_row; ++face_col) {
            const uint32_t face_idx = face_row * faces_per_row + face_col;
            volatile tt_l1_ptr uint32_t* face_ptr = ptr + face_idx * face_size_u32;

            uint32_t cols_in_face = 0;
            if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
                constexpr uint32_t cols_per_face = tt::constants::FACE_WIDTH;
                const uint32_t face_col_start = face_col * cols_per_face;
                if (valid_reduce_dim_elements_in_tile > face_col_start) {
                    const uint32_t remaining = valid_reduce_dim_elements_in_tile - face_col_start;
                    cols_in_face = remaining < cols_per_face ? remaining : cols_per_face;
                }
            } else {
                constexpr uint32_t rows_per_face = tt::constants::FACE_HEIGHT;
                const uint32_t face_row_start = face_row * rows_per_face;
                if (valid_reduce_dim_elements_in_tile > face_row_start) {
                    const uint32_t remaining = valid_reduce_dim_elements_in_tile - face_row_start;
                    cols_in_face = remaining < rows_per_face ? remaining : rows_per_face;
                }
            }

            if (cols_in_face > 0) {
                fill_face_row0_cols<data_format>(face_ptr, scaler, cols_in_face);
            }
        }
    }
}

// =============================================================================
// Format-aware fill_tile_col0 (matmul-based reduce — column 0 of left-side faces)
// =============================================================================

template <DataFormat data_format, uint32_t face_rows, uint32_t faces_per_row>
FORCE_INLINE void fill_tile_col0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_tile_col0 only supports Float16_b (bfloat16) and Float32 formats");

    constexpr uint32_t face_size_u32 =
        (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t ROWS_PER_FACE = 16;

    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        const uint32_t face_idx = face_row * faces_per_row;
        volatile tt_l1_ptr uint32_t* face_ptr = ptr + face_idx * face_size_u32;
        fill_face_col0_rows<data_format>(face_ptr, scaler, ROWS_PER_FACE);
    }
}

// =============================================================================
// Format-aware fill_tile_col0_partial — fills only first valid rows of column 0
// =============================================================================

template <DataFormat data_format, uint32_t face_rows, uint32_t faces_per_row>
FORCE_INLINE void fill_tile_col0_partial(
    volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler, uint32_t valid_reduce_dim_elements_in_tile) {
    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "fill_tile_col0_partial only supports Float16_b (bfloat16) and Float32 formats");

    constexpr uint32_t face_size_u32 =
        (data_format == DataFormat::Float32) ? FACE_SIZE_U32_FP32 : FACE_SIZE_U32;
    constexpr uint32_t ROWS_PER_FACE = 16;

    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        const uint32_t face_row_start = face_row * ROWS_PER_FACE;
        uint32_t rows_in_face = 0;
        if (valid_reduce_dim_elements_in_tile > face_row_start) {
            const uint32_t remaining = valid_reduce_dim_elements_in_tile - face_row_start;
            rows_in_face = remaining < ROWS_PER_FACE ? remaining : ROWS_PER_FACE;
        }

        if (rows_in_face > 0) {
            const uint32_t face_idx = face_row * faces_per_row;
            volatile tt_l1_ptr uint32_t* face_ptr = ptr + face_idx * face_size_u32;
            fill_face_col0_rows<data_format>(face_ptr, scaler, rows_in_face);
        }
    }
}

// =============================================================================
// Prepare CB tile for reduce using a caller-provided float scaler
// =============================================================================

template <uint32_t cb_id, PoolType pool_type, ReduceDim reduce_dim, bool compute_uses_reduce_tile>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f, uint32_t valid_reduce_dim_elements_in_tile) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    constexpr DataFormat data_format = get_dataformat(cb_id);
    constexpr uint32_t tile_r_dim = get_tile_r_dim<cb_id>();
    constexpr uint32_t tile_c_dim = get_tile_c_dim<cb_id>();
    static_assert(tile_r_dim % tt::constants::FACE_HEIGHT == 0, "tile height must be a multiple of FACE_HEIGHT");
    static_assert(tile_c_dim % tt::constants::FACE_WIDTH == 0, "tile width must be a multiple of FACE_WIDTH");
    constexpr uint32_t face_rows = tile_r_dim / tt::constants::FACE_HEIGHT;
    constexpr uint32_t faces_per_row = tile_c_dim / tt::constants::FACE_WIDTH;
    constexpr uint32_t num_faces = face_rows * faces_per_row;
    static_assert(
        reduce_dim != ReduceDim::REDUCE_SCALAR
            || (tile_r_dim == tt::constants::TILE_HEIGHT && tile_c_dim == tt::constants::TILE_WIDTH),
        "REDUCE_SCALAR only supports full 32x32 tiles");

    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "prepare_reduce_scaler only supports Float16_b (bfloat16) and Float32 formats");

    ASSERT(valid_reduce_dim_elements_in_tile > 0);

    // Matmul-based reduce uses col-0 fill; reduce LLK uses row-0 fill
    constexpr bool use_matmul = !compute_uses_reduce_tile && reduce_uses_matmul<pool_type, reduce_dim>();

    experimental::CircularBuffer cb(cb_id);

    cb.reserve_back(1);
    uint32_t write_addr = cb.get_write_ptr();

    zero_tile<cb_id>(write_addr);

    if constexpr (use_matmul) {
        uint32_t scaler = float_to_col0_scaler_bits<data_format>(scaler_f);
        if (scaler != 0) {
            if (valid_reduce_dim_elements_in_tile == tile_r_dim) {
                fill_tile_col0<data_format, face_rows, faces_per_row>(addr_to_l1_ptr(write_addr), scaler);
            } else {
                fill_tile_col0_partial<data_format, face_rows, faces_per_row>(
                    addr_to_l1_ptr(write_addr), scaler, valid_reduce_dim_elements_in_tile);
            }
        }
    } else {
        uint32_t scaler = float_to_scaler_bits<data_format>(scaler_f);
        if (scaler != 0) {
            if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
                fill_each_face_row0<data_format, num_faces>(addr_to_l1_ptr(write_addr), scaler);
            } else {
                constexpr uint32_t full_dim =
                    (reduce_dim == ReduceDim::REDUCE_COL) ? tile_r_dim : tile_c_dim;
                if (valid_reduce_dim_elements_in_tile == full_dim) {
                    fill_each_face_row0<data_format, num_faces>(addr_to_l1_ptr(write_addr), scaler);
                } else {
                    fill_each_face_row0_partial<data_format, reduce_dim, face_rows, faces_per_row>(
                        addr_to_l1_ptr(write_addr), scaler, valid_reduce_dim_elements_in_tile);
                }
            }
        }
    }

    cb.push_back(1);
}

// =============================================================================
// Format-aware calculate_and_prepare_reduce_scaler (cb_id-deduced format and tile shape)
// =============================================================================

template <
    uint32_t cb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t reduce_factor,
    bool compute_uses_reduce_tile>
FORCE_INLINE void calculate_and_prepare_reduce_scaler(uint32_t valid_reduce_dim_elements_in_tile) {
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
    prepare_reduce_scaler<cb_id, pool_type, reduce_dim, compute_uses_reduce_tile>(scaler_f, valid_reduce_dim_elements_in_tile);
}

}  // namespace dataflow_kernel_lib
