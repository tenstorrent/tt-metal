// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Writer for per_token_cast_to_fp8 (v0).
// Reads bf16/fp32 sticks from cb_in_rm, converts to fp8 row-major in software, and writes to DRAM.
// Also writes 1.0f constants to the scale tensor for every stick.
//
// Conversion is done in this kernel (RISC-V software) rather than via the compute kernel because
// tt-metal's LLK does not currently expose a reliable bf16/fp32 -> fp8 path that we can compose with
// tilize. The combine/dispatch ops use pack_untilize for that, but they require tile-layout input,
// not the row-major input we accept.
//
// FP8 E4M3 conversion: round-to-nearest with clipping to ±448 (max normal). Denormals flush to 0.

inline uint8_t bf16_to_fp8(uint16_t bf16) {
    uint16_t sign = bf16 >> 15;
    int32_t bf16_exp = (bf16 >> 7) & 0xFF;
    uint16_t bf16_mant_high = (bf16 >> 4) & 0x7;  // top 3 bits of mantissa (going into fp8)
    uint16_t bf16_round_bit = (bf16 >> 3) & 0x1;  // round bit
    uint16_t bf16_sticky = (bf16 & 0x7);          // sticky bits

    if (bf16_exp == 0) {
        // bf16 zero or denormal → flush to fp8 ±0
        return (uint8_t)(sign << 7);
    }
    if (bf16_exp == 0xFF) {
        // bf16 inf/NaN → fp8 max normal (clipping behaviour matches DeepEP cast)
        return (uint8_t)(0x7E | (sign << 7));
    }

    // fp8 exponent = bf16_exp - 120 (= bf16_exp - 127 + 7)
    int32_t fp8_exp = bf16_exp - 120;

    if (fp8_exp >= 15) {
        // Overflow: in e4m3, exp=15 with mantissa=7 is NaN. Saturate at S.1111.110 = ±448.
        return (uint8_t)(0x7E | (sign << 7));
    }
    if (fp8_exp <= 0) {
        // Underflow into denormal range.
        // bf16 normal value = 1.bf16_mant * 2^(bf16_exp - 127)
        // fp8 denormal value = (m/8) * 2^-6 where m ∈ {1..7}
        // Need 1.bf16_mant * 2^(bf16_exp - 127) = (m/8) * 2^-6
        //   → 1.bf16_mant * 2^(bf16_exp - 121) = m/8
        //   → m = 8 * 1.bf16_mant * 2^(bf16_exp - 121)
        // For fp8_exp <= 0, bf16_exp <= 120, so 2^(bf16_exp - 121) ≤ 2^-1.
        // Shift right by (1 - fp8_exp) the mantissa with implicit leading 1.
        // shift = 1 - fp8_exp  ∈ [1, ∞)
        int32_t shift = 1 - fp8_exp;  // ≥ 1
        if (shift > 4) {
            // Too small to represent even as denormal → flush to ±0
            return (uint8_t)(sign << 7);
        }
        uint32_t implicit_one_and_mant = (1u << 3) | bf16_mant_high;  // 4-bit value, leading bit = 1
        // Round-to-nearest: get the bit shifted out
        uint32_t pre_round = implicit_one_and_mant >> (shift - 1);  // includes future round bit
        uint8_t m = (uint8_t)((pre_round >> 1) & 0x7);
        uint8_t round_bit = (uint8_t)(pre_round & 1);
        // sticky from shifted-out tail
        uint32_t shifted_out_mask = (1u << (shift - 1)) - 1;
        uint8_t sticky = ((implicit_one_and_mant & shifted_out_mask) != 0 || bf16_round_bit || bf16_sticky) ? 1 : 0;
        uint8_t lsb = m & 1;
        if (round_bit && (sticky || lsb)) {
            m++;
            if (m == 8) {
                // Round-up into the smallest normal: S.0001.000
                return (uint8_t)((sign << 7) | (1 << 3));
            }
        }
        if (m == 0) {
            return (uint8_t)(sign << 7);  // ±0 (denormal rounded down)
        }
        return (uint8_t)((sign << 7) | m);
    }

    // Normal-to-normal conversion. fp8 mantissa = top 3 bits of bf16 mantissa, with round-to-nearest.
    uint8_t m = (uint8_t)bf16_mant_high;
    uint8_t round_bit = (uint8_t)bf16_round_bit;
    uint8_t sticky = (bf16_sticky != 0) ? 1 : 0;
    uint8_t lsb = m & 1;
    if (round_bit && (sticky || lsb)) {
        m++;
        if (m == 8) {
            m = 0;
            fp8_exp++;
            if (fp8_exp >= 15) {
                return (uint8_t)(0x7E | (sign << 7));
            }
        }
    }
    // Avoid producing the NaN encoding S.1111.111
    if (fp8_exp == 15 && m == 7) {
        m = 6;
    }
    return (uint8_t)((sign << 7) | ((uint8_t)fp8_exp << 3) | m);
}

inline uint8_t fp32_to_fp8(uint32_t fp32) {
    // Round bf16 value first (drop lower 16 bits with round-to-nearest-even), then bf16 → fp8.
    uint16_t bf16 = (uint16_t)((fp32 + ((fp32 >> 16) & 1) + 0x7FFF) >> 16);
    return bf16_to_fp8(bf16);
}

void kernel_main() {
    uint32_t dst_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_scale_addr = get_arg_val<uint32_t>(1);
    uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch_fp8 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale_const = get_compile_time_arg_val(2);
    constexpr uint32_t in_element_size = get_compile_time_arg_val(3);  // 2 (bf16) or 4 (fp32)
    constexpr uint32_t in_stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t e4m3_stick_size_bytes = get_compile_time_arg_val(5);  // = H
    constexpr uint32_t scale_write_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t is_fp32 = get_compile_time_arg_val(7);
    constexpr uint32_t H = get_compile_time_arg_val(8);
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr auto e4m3_args = TensorAccessorArgs<9>();
    constexpr auto scale_args = TensorAccessorArgs<e4m3_args.next_compile_time_args_offset()>();

    const auto e4m3_dst = TensorAccessor(e4m3_args, dst_e4m3_addr);
    const auto scale_dst = TensorAccessor(scale_args, dst_scale_addr);

    // Pre-fill scale constants L1 region with 1.0f for every element.
    constexpr uint32_t ONE_F32_BITS = 0x3f800000u;
    constexpr uint32_t scale_write_elements = scale_write_size_bytes / 4;
    const uint32_t scale_const_l1 = get_write_ptr(cb_scale_const);
    {
        volatile tt_l1_ptr uint32_t* scale_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scale_const_l1);
        for (uint32_t i = 0; i < scale_write_elements; ++i) {
            scale_buf[i] = ONE_F32_BITS;
        }
    }

    // L1 scratch region for one tile-row of converted fp8 bytes (TILE_HEIGHT * H bytes).
    const uint32_t fp8_scratch_l1 = get_write_ptr(cb_scratch_fp8);

    for (uint32_t row = 0; row < num_tile_rows; ++row) {
        cb_wait_front(cb_in_rm, TILE_HEIGHT);
        const uint32_t in_l1 = get_read_ptr(cb_in_rm);

        if constexpr (is_fp32) {
            const volatile tt_l1_ptr uint32_t* in_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(in_l1);
            volatile tt_l1_ptr uint8_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(fp8_scratch_l1);
            for (uint32_t i = 0; i < TILE_HEIGHT * H; ++i) {
                out_ptr[i] = fp32_to_fp8(in_ptr[i]);
            }
        } else {
            const volatile tt_l1_ptr uint16_t* in_ptr = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(in_l1);
            volatile tt_l1_ptr uint8_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(fp8_scratch_l1);
            for (uint32_t i = 0; i < TILE_HEIGHT * H; ++i) {
                out_ptr[i] = bf16_to_fp8(in_ptr[i]);
            }
        }

        // Write 32 e4m3 sticks to DRAM.
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint32_t stick_id = (start_tile_row + row) * TILE_HEIGHT + s;
            noc_async_write(
                fp8_scratch_l1 + s * e4m3_stick_size_bytes, e4m3_dst.get_noc_addr(stick_id), e4m3_stick_size_bytes);
        }

        // Write 32 scale sticks of 1.0f constants.
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint32_t stick_id = (start_tile_row + row) * TILE_HEIGHT + s;
            noc_async_write(scale_const_l1, scale_dst.get_noc_addr(stick_id), scale_write_size_bytes);
        }

        noc_async_write_barrier();
        cb_pop_front(cb_in_rm, TILE_HEIGHT);
    }
}
