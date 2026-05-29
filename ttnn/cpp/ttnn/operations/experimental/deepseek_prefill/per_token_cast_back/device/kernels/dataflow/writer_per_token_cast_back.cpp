// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Writer for per_token_cast_back (v0, ignores scale).
// Reads FP8_E4M3 bytes from cb_in_fp8, converts to bf16/fp32 in software (lookup table built at
// kernel startup), and writes ROW_MAJOR sticks to DRAM. Avoiding the compute path entirely because
// tt-metal's LLK does not currently support fp8 ROW_MAJOR -> bf16/fp32 via the unpacker.
//
// Compile-time args:
//   0: cb_in_fp8     (CB index of incoming fp8 row-major sticks)
//   1: cb_scratch    (CB index of small L1 scratch buffer for converted bf16/fp32 stick)
//   2: stick_size_fp8   (= H, in bytes)
//   3: out_element_size (2 for bf16, 4 for fp32)
//   4: out_stick_size   (= H * out_element_size)
//   5: is_fp32       (0 = bf16, 1 = fp32)
//   6..: TensorAccessorArgs for the output buffer

// Build the FP8 -> BF16 lookup table at kernel startup. Idempotent.
inline uint16_t fp8_to_bf16(uint8_t fp8) {
    uint16_t sign = static_cast<uint16_t>(fp8 >> 7);
    uint16_t exp = static_cast<uint16_t>((fp8 >> 3) & 0xF);
    uint16_t mantissa = static_cast<uint16_t>(fp8 & 0x7);
    if (exp == 0) {
        if (mantissa == 0) {
            return sign << 15;  // ±0
        }
        // Denormal: normalize. Find leading 1 in mantissa (3 bits).
        uint16_t lz;
        uint16_t m;
        if (mantissa & 0x4) {
            lz = 0;
            m = mantissa & 0x3;
        } else if (mantissa & 0x2) {
            lz = 1;
            m = (mantissa << 1) & 0x6;
        } else {  // mantissa == 1
            lz = 2;
            m = 0;
        }
        uint16_t bf16_exp = 127 - 6 - lz;
        uint16_t bf16_mantissa = m << 4;
        return (sign << 15) | (bf16_exp << 7) | bf16_mantissa;
    } else if (exp == 0xF && mantissa == 0x7) {
        // FP8 E4M3 NaN (S.1111.111)
        return 0x7FC0 | (sign << 15);
    } else {
        uint16_t bf16_exp = exp + 120;  // 127 - 7 = 120
        uint16_t bf16_mantissa = mantissa << 4;
        return (sign << 15) | (bf16_exp << 7) | bf16_mantissa;
    }
}

inline uint32_t fp8_to_fp32(uint8_t fp8) {
    uint32_t bf16 = fp8_to_bf16(fp8);
    // bf16 is the top 16 bits of the equivalent fp32.
    return bf16 << 16;
}

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in_fp8 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_fp8 = get_compile_time_arg_val(2);
    constexpr uint32_t out_element_size = get_compile_time_arg_val(3);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t is_fp32 = get_compile_time_arg_val(5);
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr auto dst_args = TensorAccessorArgs<6>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    // Get a writable L1 region (cb_scratch holds TILE_HEIGHT sticks worth of converted output).
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);

    for (uint32_t row = 0; row < num_tile_rows; ++row) {
        cb_wait_front(cb_in_fp8, TILE_HEIGHT);
        const uint32_t fp8_l1 = get_read_ptr(cb_in_fp8);
        const volatile tt_l1_ptr uint8_t* fp8_ptr = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(fp8_l1);

        if constexpr (is_fp32) {
            volatile tt_l1_ptr uint32_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
            for (uint32_t i = 0; i < TILE_HEIGHT * stick_size_fp8; ++i) {
                out_ptr[i] = fp8_to_fp32(fp8_ptr[i]);
            }
        } else {
            volatile tt_l1_ptr uint16_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_l1);
            for (uint32_t i = 0; i < TILE_HEIGHT * stick_size_fp8; ++i) {
                out_ptr[i] = fp8_to_bf16(fp8_ptr[i]);
            }
        }

        // Write 32 converted sticks to DRAM.
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint32_t stick_id = (start_tile_row + row) * TILE_HEIGHT + s;
            noc_async_write(scratch_l1 + s * out_stick_size, dst.get_noc_addr(stick_id), out_stick_size);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_in_fp8, TILE_HEIGHT);
    }
}
