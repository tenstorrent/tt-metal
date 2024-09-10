// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#ifdef DEBUG_PRINT
// this function is useful for printing bfloat16 values
#include "dprint.h"

float bfloat16_to_float32(uint16_t bfloat16_data) {
    uint32_t bits = static_cast<uint32_t>(bfloat16_data) << 16;

    // Extract the sign bit
    uint32_t sign = bits & 0x80000000;

    // Extract the exponent
    uint32_t exponent = bits & 0x7F800000;

    // Extract the mantissa
    uint32_t mantissa = bits & 0x007FFFFF;

    // Handle special cases
    if (exponent == 0 && mantissa == 0) {
        // Zero
        return sign ? -0.0f : 0.0f;
    } else if (exponent == 0x7F800000) {
        if (mantissa == 0) {
            // Infinity
            return sign ? -__builtin_huge_valf() : __builtin_huge_valf();
        } else {
            // NaN
            return __builtin_nanf("");
        }
    }

    // Assemble the float
    union {
        uint32_t u;
        float f;
    } ieee_float;

    ieee_float.u = sign | exponent | mantissa;
    return ieee_float.f;
}
#endif


void kernel_main() {

    constexpr bool src0_is_dram          = (bool) get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);

    constexpr uint32_t stride_W = get_compile_time_arg_val(5);
    constexpr uint32_t stride_H = get_compile_time_arg_val(6);
    constexpr uint32_t stride_C = get_compile_time_arg_val(7);
    constexpr uint32_t stride_N = get_compile_time_arg_val(8);
    constexpr uint32_t page_size = get_compile_time_arg_val(9);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_W = get_arg_val<uint32_t>(1);
    const uint32_t start_H = get_arg_val<uint32_t>(2);
    const uint32_t start_C = get_arg_val<uint32_t>(3);
    const uint32_t start_N = get_arg_val<uint32_t>(4);

    const uint32_t end_W = get_arg_val<uint32_t>(5);
    const uint32_t end_H = get_arg_val<uint32_t>(6);
    const uint32_t end_C = get_arg_val<uint32_t>(7);
    const uint32_t end_N = get_arg_val<uint32_t>(8);

    const InterleavedAddrGen<src0_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = page_size
    };

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_out0 = 24;
    uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* in_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_buffer_l1_addr);
    constexpr uint32_t CH = C*H;
    // TODO: optimize this kernel to read in multiple sticks at once
    // TODO: add support for negative strides
    // TODO: add axis support
    for (uint32_t i = start_N; i < end_N; i+=stride_N) {
        uint32_t iCH = i*CH;
        for (uint32_t j = start_C; j < end_C; j+=stride_C) {
            uint32_t jHplusiCH = j*H + iCH;
            for (uint32_t k = start_H; k < end_H; k+=stride_H) {

                // relevant page/stick id
                uint32_t src_stick_id = k + jHplusiCH;

                // read in entire stick and wait - we may want to allocate a CB and max out our reads before waiting
                noc_async_read_page(src_stick_id, s0, src_buffer_l1_addr);
                noc_async_read_barrier();


                // TODO: optimize when there's no slice or stride along W. In that case, we can just do a single read and write.
                // reserve space in output buffer
                cb_reserve_back(cb_id_out0, 1);
                // write out element by element into output buffer
                volatile tt_l1_ptr uint16_t* out_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id_out0));
                uint32_t out_stick_id = 0;
                for (uint32_t l = start_W; l < end_W; l+=stride_W) {
                    out_stick[out_stick_id] = in_stick[l];
                    out_stick_id++;
                }
                cb_push_back(cb_id_out0, 1);
            }
        }
    }


}
