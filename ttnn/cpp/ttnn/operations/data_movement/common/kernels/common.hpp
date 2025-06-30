// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common kernel functions used in data movement device kernels
// It's best to copy and paste the functions in rather than include the header as code size will likely explode
// Best to separate in to cpp/hpp at some point to avoid the code size explosion but need to figure out the linking
// issues
#include <stdio.h>
#include <cstring>
#include <type_traits>

constexpr uint64_t ALIGN_REQ_64 = 64;
constexpr uint64_t MASK_64 = 0xFFFFFFFFFFFFFFC0;
constexpr uint64_t OFFSET_64 = 0x000000000000003F;
constexpr uint64_t ALIGN_REQ_16 = 16;
constexpr uint64_t MASK_16 = 0xFFFFFFFFFFFFFFF0;
constexpr uint64_t OFFSET_16 = 0x000000000000000F;

namespace tt::data_movement::common {

template <uint32_t max_transfer_size, bool only_reads>
FORCE_INLINE void enhanced_noc_async_read(
    const uint64_t src_noc_addr, const uint32_t dst_l1_addr, const uint32_t bytes) {
    // If you do not know the max_transfer_size at compile time write 0 to it.
    // only reads is true if we ONLY use noc_async_read and all calls to tt_memmove have use_read_datamover as True
    if constexpr (only_reads && max_transfer_size <= NOC_MAX_BURST_SIZE) {
        noc_async_read_one_packet(src_noc_addr, dst_l1_addr, bytes);
    } else {
        noc_async_read<max_transfer_size == 0 ? NOC_MAX_BURST_SIZE + 1 : max_transfer_size>(
            src_noc_addr, dst_l1_addr, bytes);
    }
}

template <uint32_t max_transfer_size, bool only_writes>
FORCE_INLINE void enhanced_noc_async_write(
    const uint32_t src_l1_addr, const uint64_t dst_noc_addr, const uint32_t bytes) {
    // If you do not know the max_transfer_size at compile time write 0 to it.
    // only writes is true if we ONLY use noc_async_read and all calls to tt_memmove have use_read_datamover as False
    if constexpr (only_writes && max_transfer_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_one_packet(src_l1_addr, dst_noc_addr, bytes);
    } else {
        noc_async_write<max_transfer_size == 0 ? NOC_MAX_BURST_SIZE + 1 : max_transfer_size>(
            src_l1_addr, dst_noc_addr, bytes);
    }
}

template <bool guaranteed_16B_aligned, bool copy_async, bool use_read_datamover, uint32_t max_transfer_size>
FORCE_INLINE void tt_memmove(const uint32_t dst_l1_addr, const uint32_t src_l1_addr, const uint32_t bytes) {
    // Function performs a memory copy between two l1 addresses in the local core
    // Uses noc_async_read when possible to copy the data over
    // Set guaranteed 16B aligned to true if the source and destination are externally guaranteed to be 16B aligned
    // (dangerous) Set copy_async to true if you wish to perform the operation asynchronously, in this case you can add
    // a noc_async_read_barrier to synchronize later
    if constexpr (use_read_datamover) {
        if constexpr (guaranteed_16B_aligned) {
            enhanced_noc_async_read<max_transfer_size, false>(get_noc_addr(src_l1_addr), dst_l1_addr, bytes);
            if constexpr (!copy_async) {
                noc_async_read_barrier();
            }
        } else {
            if ((dst_l1_addr & OFFSET_16) == (src_l1_addr & OFFSET_16)) {
                enhanced_noc_async_read<max_transfer_size, false>(get_noc_addr(src_l1_addr), dst_l1_addr, bytes);
                if constexpr (!copy_async) {
                    noc_async_read_barrier();
                }
            } else {
                memmove((void*)(dst_l1_addr), (void*)(src_l1_addr), (size_t)(bytes));
            }
        }
    } else {
        if constexpr (guaranteed_16B_aligned) {
            enhanced_noc_async_write<max_transfer_size, false>(src_l1_addr, get_noc_addr(dst_l1_addr), bytes);
            if constexpr (!copy_async) {
                noc_async_write_barrier();
            }
        } else {
            if ((dst_l1_addr & OFFSET_16) == (src_l1_addr & OFFSET_16)) {
                enhanced_noc_async_write<max_transfer_size, false>(src_l1_addr, get_noc_addr(dst_l1_addr), bytes);
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

// Function template to swap two elements in a uint32_t array
template <size_t N>
FORCE_INLINE void swap_elements(uint32_t (&array)[N], size_t i, size_t j) {
    // Perform the swap
    uint32_t temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

// 2D Transpose function for debug use in reader/writer kernels
FORCE_INLINE void transpose_2d(
    uint32_t input_l1_addr,
    uint32_t output_l1_addr,
    uint32_t X,
    uint32_t W,
    uint32_t element_size,
    uint32_t input_page_size,
    uint32_t output_page_size) {
    volatile tt_l1_ptr uint8_t* input_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(input_l1_addr);
    volatile tt_l1_ptr uint8_t* output_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(output_l1_addr);
    // transpose from XW, where X is outer and W inner, to WX, where W is outer and X is inner
    // each element is element_size bytes
    // each row is W elements, and each row is separated by input_page_size bytes
    // each output row is X elements, and each row is separated by output_page_size bytes

    for (uint32_t x = 0; x < X; ++x) {
        for (uint32_t w = 0; w < W; ++w) {
            // Compute the input and output addresses
            uint32_t input_addr = x * input_page_size + w * element_size;
            uint32_t output_addr = w * output_page_size + x * element_size;
            // Copy the element - do we have memcpy? use this for now
            for (uint32_t i = 0; i < element_size; ++i) {
                output_ptr[output_addr + i] = input_ptr[input_addr + i];
            }
        }
    }
}

template <uint32_t AlignReq>
FORCE_INLINE uint32_t align_address(const uint32_t address, const uint64_t mask) {
    return (address & mask) + AlignReq;
}

// Wait for a specified number of cycles
// This is a blocking wait, so it should only be used for debugging purposes
// It is not recommended to use this in production code
inline void spin(uint32_t cycles) {
    volatile uint tt_reg_ptr* clock_lo = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr* clock_hi = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint64_t wall_clock_timestamp = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    uint64_t wall_clock = 0;
    do {
        wall_clock = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    } while (wall_clock < (wall_clock_timestamp + cycles));
}

}  // namespace tt::data_movement::common
