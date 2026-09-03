// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for reduce_helpers_dataflow.hpp
// Do not include directly - include reduce_helpers_dataflow.hpp instead

#include <cmath>
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
// Prepare CB tile for reduce using a caller-provided float scaler
// =============================================================================

template <uint32_t dfb_id, PoolType pool_type, ReduceDim reduce_dim>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f, uint32_t valid_reduce_dim_elements_in_tile) {
    constexpr DataFormat data_format = get_dataformat(dfb_id);
    constexpr uint32_t tile_r_dim = get_tile_r_dim<dfb_id>();
    constexpr uint32_t tile_c_dim = get_tile_c_dim<dfb_id>();
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

    constexpr uint32_t full_dim = (reduce_dim == ReduceDim::REDUCE_COL) ? tile_r_dim : tile_c_dim;

    DataflowBuffer dfb(dfb_id);

    dfb.reserve_back(1);
    uint32_t write_addr = dfb.get_write_ptr();

    Noc noc;
    noc.async_write_zeros(dfb, get_tile_size(dfb_id));
    noc.write_zeros_l1_barrier();

    uint32_t scaler = float_to_scaler_bits<data_format>(scaler_f);
    if (scaler != 0) {
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
            fill_each_face_row0<data_format, num_faces>(addr_to_l1_ptr(write_addr), scaler);
        } else {
            if (valid_reduce_dim_elements_in_tile == full_dim) {
                fill_each_face_row0<data_format, num_faces>(addr_to_l1_ptr(write_addr), scaler);
            } else {
                fill_each_face_row0_partial<data_format, reduce_dim, face_rows, faces_per_row>(
                    addr_to_l1_ptr(write_addr), scaler, valid_reduce_dim_elements_in_tile);
            }
        }
    }

    // Quasar DM cores have a write-back L1 D-cache (4KB, per-core) + L2 cache
    // (128KB, shared between DM cores). RISC stores flow Core -> L1 D$ -> L2 -> TL1
    // (Tensix L1, the SRAM that other RISCs read). The volatile fills above land
    // in DM-private caches; without an explicit flush, TRISC-side unpack reads
    // stale TL1 contents (zeros) even though the DM observes its own writes via
    // L1 D$ hits. This is consistent with the runtime evidence:
    //   DM:    PRS:after_fill @0xbb780 : 0x3f803f80 ...
    //   UNPACK U:scaler sh@0xbb780     : 0x0 0x0 ...
    // For NoC-written input tiles the NoC engine writes directly to TL1 and
    // bypasses DM caches, which is why those are visible to TRISC.
    // flush_l2_cache_range probes L1 D$ for dirty data and writes through to TL1,
    // so TRISC sees the freshly-filled scaler tile once we signal push_back.
    // On non-Quasar (or non-DM) builds this is a no-op.
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    {
        constexpr uint32_t tile_size_bytes = get_tile_size(cb_id);
        flush_l2_cache_range(write_addr, tile_size_bytes);
    }
#endif
    dfb.push_back(1);
}

// =============================================================================
// Format-aware calculate_and_prepare_reduce_scaler (dfb_id-deduced format and tile shape)
// =============================================================================

template <
    uint32_t dfb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t reduce_factor>
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
    // 2. Fill the DFB with the computed scaler
    // -------------------------------------------------------------------------
    prepare_reduce_scaler<dfb_id, pool_type, reduce_dim>(scaler_f, valid_reduce_dim_elements_in_tile);
}

}  // namespace dataflow_kernel_lib
