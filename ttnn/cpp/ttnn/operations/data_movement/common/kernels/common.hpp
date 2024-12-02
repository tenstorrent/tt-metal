// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common kernel functions used in data movement device kernels
// It's best to copy and paste the functions in rather than include the header as code size will likely explode
// Best to separate in to cpp/hpp at some point to avoid the code size explosion but need to figure out the linking
// issues
#include <stdio.h>
#include <cstring>
#define MASK_64      0xFFFFFFFFFFFFFFC0
#define OFFSET_64    0x000000000000003F
#define MASK_16      0xFFFFFFFFFFFFFFF0
#define OFFSET_16    0x000000000000000F

namespace tt::data_movement::common {

template <bool guaranteed_16B_alligned, bool copy_async, bool use_read_datamover>
FORCE_INLINE void tt_memmove(const uint32_t dst_l1_addr, const uint32_t src_l1_addr, const uint32_t bytes) {
    //Function performs a memory copy between two l1 addresses in the local core
    //Uses noc_async_read when possible to copy the data over
    //Set guaranteed 16B alligned to true if the source and destination are externally guaranteed to be 16B alligned (dangerous)
    //Set copy_async to true if you wish to perform the operation asynchronously, in this case you can add a noc_async_read_barrier to synchronize later
    if constexpr (use_read_datamover) {
        if constexpr (guaranteed_16B_alligned) {
            noc_async_read(get_noc_addr(src_l1_addr), dst_l1_addr, bytes);
            if constexpr (!copy_async) {
                noc_async_read_barrier();
            }
        } else {
            if ((dst_l1_addr & OFFSET_16) == (src_l1_addr & OFFSET_16)) {
                noc_async_read(get_noc_addr(src_l1_addr), dst_l1_addr, bytes);
                if constexpr (!copy_async) {
                    noc_async_read_barrier();
                }
            } else {
                memmove((void*)(dst_l1_addr), (void*)(src_l1_addr), (size_t)(bytes));
            }
        }
    } else {
        if constexpr (guaranteed_16B_alligned) {
            noc_async_write(src_l1_addr, get_noc_addr(dst_l1_addr), bytes);
            if constexpr (!copy_async) {
                noc_async_write_barrier();
            }
        } else {
            if ((dst_l1_addr & OFFSET_16) == (src_l1_addr & OFFSET_16)) {
                noc_async_write(src_l1_addr, get_noc_addr(dst_l1_addr), bytes);
                if constexpr (!copy_async) {
                    noc_async_write_barrier();
                }
            } else {
                memmove((void*)(dst_l1_addr), (void*)(src_l1_addr), (size_t)(bytes));
            }
        }
    }
}

// this function is useful for converting bfloat16 values to float32
FORCE_INLINE float bfloat16_to_float32(uint16_t bfloat16_data) {
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

FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, uint32_t val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

// Utility functions
template <uint32_t a, uint32_t b>
FORCE_INLINE constexpr uint32_t div_up() {
    static_assert(b > 0, "divisor must be greater than 0");
    return static_cast<uint32_t>((a + b - 1) / b);
}

template <uint32_t a, uint32_t b>
FORCE_INLINE constexpr uint32_t round_up() {
    return b * div_up<a, b>();
}
}  // namespace tt::data_movement::common
