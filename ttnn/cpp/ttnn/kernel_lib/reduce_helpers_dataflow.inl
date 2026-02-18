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
// Legacy fill_row0 (bfloat16 only)
// =============================================================================

template <bool half_tile>
FORCE_INLINE void fill_row0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    constexpr uint32_t num_faces = half_tile ? 2 : 4;

    for (uint32_t face = 0; face < num_faces; ++face) {
        uint32_t face_offset = face * FACE_SIZE_U32;
        for (uint32_t column = 0; column < ROW_SIZE_U32; ++column) {
            ptr[face_offset + column] = scaler;
        }
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
// Legacy calculate_and_prepare_reduce_scaler (pre-computed scaler)
// =============================================================================

template <bool half_tile>
FORCE_INLINE void calculate_and_prepare_reduce_scaler_legacy(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    // Verify scaler is properly packed: high 16 bits must equal low 16 bits
    ASSERT((scaler >> 16) == (scaler & 0xFFFF));

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<half_tile>(write_addr);

    if (scaler != 0) {
        fill_row0<half_tile>(addr_to_l1_ptr(write_addr), scaler);
    }

    cb_push_back(cb_id, 1);
}

// =============================================================================
// Prepare CB tile for reduce using a caller-provided float scaler
// =============================================================================

template <uint32_t cb_id>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);

    constexpr DataFormat data_format = get_dataformat(cb_id);
    constexpr bool half_tile = get_tile_num_faces(cb_id) == 2;

    static_assert(
        data_format == DataFormat::Float16_b || data_format == DataFormat::Float32,
        "prepare_reduce_scaler only supports Float16_b (bfloat16) and Float32 formats");

    uint32_t scaler = float_to_scaler_bits<data_format>(scaler_f);

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<data_format, half_tile>(write_addr);

    if (scaler != 0) {
        fill_row0<data_format, half_tile>(addr_to_l1_ptr(write_addr), scaler);
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
    uint32_t reduce_volume>
FORCE_INLINE void calculate_and_prepare_reduce_scaler(const float input_scaler) {

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
            static_assert(reduce_volume > 0, "reduce_volume must be greater than 0");
            scaler_f = 1.0f / sqrtf(static_cast<float>(reduce_volume));
        } else {
            scaler_f = 1.0f / static_cast<float>(reduce_volume);
        }
    } else {
        scaler_f = 1.0f;
    }

    // -------------------------------------------------------------------------
    // 1b. Apply input_scaler multiplier
    //     For REDUCE_SCALAR, sqrt the input_scaler since the LLK squares it
    // -------------------------------------------------------------------------
    scaler_f *= input_scaler;

    // -------------------------------------------------------------------------
    // 2. Fill the CB with the computed scaler
    // -------------------------------------------------------------------------
    prepare_reduce_scaler<cb_id>(scaler_f);
}

}  // namespace dataflow_kernel_lib
