// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Bicubic upsample reader kernel.
// Reads 16 neighbors in 4 groups of 4, computes cubic weights using Q16.16
// fixed-point arithmetic, pushes data + weights to CBs for Tensix compute.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>

using namespace fixed_point_arithmetic;

// Signed fixed-point to BF16 — handles negative values (cubic weights can be negative)
// The library's fixed_to_bf16 treats input as unsigned; this wraps it with sign handling.
inline uint16_t signed_fixed_to_bf16(int32_t fixed_val) {
    if (fixed_val == 0) {
        return 0x0000;
    }
    if (fixed_val > 0) {
        return fixed_to_bf16(fixed_val);
    }
    // Negative: convert magnitude, set sign bit (bit 15 of BF16)
    uint16_t magnitude = fixed_to_bf16(-fixed_val);
    return magnitude | 0x8000;
}

// Q16.16 constants for cubic kernel (a = -0.5)
// W(t) = 1.5|t|^3 - 2.5|t|^2 + 1        for 0 <= |t| < 1
// W(t) = -0.5|t|^3 + 2.5|t|^2 - 4|t| + 2  for 1 <= |t| < 2
constexpr int32_t FIXED_THREE_HALF = (3 * FIXED_ONE) / 2;  // 1.5
constexpr int32_t FIXED_FIVE_HALF = (5 * FIXED_ONE) / 2;   // 2.5
constexpr int32_t FIXED_NEG_HALF = -(FIXED_ONE / 2);       // -0.5
constexpr int32_t FIXED_FOUR = 4 * FIXED_ONE;              // 4.0
constexpr int32_t FIXED_TWO = 2 * FIXED_ONE;               // 2.0

// Cubic weight in Q16.16 fixed-point
inline int32_t cubic_weight_fixed(int32_t t) {
    int32_t abs_t = (t < 0) ? -t : t;

    if (abs_t < FIXED_ONE) {
        // W = 1.5*|t|^3 - 2.5*|t|^2 + 1
        int32_t t2 = fixed_mul(abs_t, abs_t);
        int32_t t3 = fixed_mul(t2, abs_t);
        return fixed_mul(FIXED_THREE_HALF, t3) - fixed_mul(FIXED_FIVE_HALF, t2) + FIXED_ONE;
    } else if (abs_t < FIXED_TWO) {
        // W = -0.5*|t|^3 + 2.5*|t|^2 - 4*|t| + 2
        int32_t t2 = fixed_mul(abs_t, abs_t);
        int32_t t3 = fixed_mul(t2, abs_t);
        return fixed_mul(FIXED_NEG_HALF, t3) + fixed_mul(FIXED_FIVE_HALF, t2) - fixed_mul(FIXED_FOUR, abs_t) +
               FIXED_TWO;
    }
    return 0;
}

inline int32_t clamp_coord(int32_t val, int32_t max_val) {
    if (val < 0) {
        return 0;
    }
    if (val > max_val) {
        return max_val;
    }
    return val;
}

void kernel_main() {
    const uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_pixels = get_arg_val<uint32_t>(1);
    const uint32_t start_pixel_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t input_height = get_compile_time_arg_val(2);
    constexpr uint32_t input_width = get_compile_time_arg_val(3);
    constexpr uint32_t output_height = get_compile_time_arg_val(4);
    constexpr uint32_t output_width = get_compile_time_arg_val(5);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t aligned_input_stick_nbytes = get_compile_time_arg_val(7);
    constexpr uint32_t blocks = get_compile_time_arg_val(8);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(9);
    // Compile-time args [10] and [11] are precomputed Q16.16 ratios from host
    constexpr int32_t ratio_h_fixed = static_cast<int32_t>(get_compile_time_arg_val(10));
    constexpr int32_t ratio_w_fixed = static_cast<int32_t>(get_compile_time_arg_val(11));

    constexpr auto src_args = TensorAccessorArgs<12>();
    const auto input_tensor_accessor = TensorAccessor(src_args, input_buffer_addr, aligned_input_stick_nbytes);

    const int32_t max_h = static_cast<int32_t>(input_height) - 1;
    const int32_t max_w = static_cast<int32_t>(input_width) - 1;

    constexpr uint32_t last_block_size_bytes = input_stick_nbytes - (blocks - 1) * input_block_size_bytes;

    uint32_t pixel_id = start_pixel_id;
    for (uint32_t p = 0; p < num_output_pixels; p++) {
        const uint32_t batch = pixel_id / (output_height * output_width);
        const uint32_t remainder = pixel_id % (output_height * output_width);
        const uint32_t y_out = remainder / output_width;
        const uint32_t x_out = remainder % output_width;

        // Coordinate mapping in Q16.16: src = (out + 0.5) * ratio - 0.5
        const int32_t src_h_fixed = fixed_mul(int_to_fixed(y_out) + FIXED_HALF, ratio_h_fixed) - FIXED_HALF;
        const int32_t src_w_fixed = fixed_mul(int_to_fixed(x_out) + FIXED_HALF, ratio_w_fixed) - FIXED_HALF;

        // Floor and fractional parts directly from Q16.16
        int32_t floor_h = src_h_fixed >> FIXED_POINT_SHIFT;
        int32_t frac_h = src_h_fixed & ((1 << FIXED_POINT_SHIFT) - 1);
        // Handle negative: if src < 0, floor should be one less, frac should be positive
        if (src_h_fixed < 0 && frac_h != 0) {
            floor_h -= 1;
            frac_h = src_h_fixed - int_to_fixed(floor_h);
        }

        int32_t floor_w = src_w_fixed >> FIXED_POINT_SHIFT;
        int32_t frac_w = src_w_fixed & ((1 << FIXED_POINT_SHIFT) - 1);
        if (src_w_fixed < 0 && frac_w != 0) {
            floor_w -= 1;
            frac_w = src_w_fixed - int_to_fixed(floor_w);
        }

        // 4 cubic weights per axis in Q16.16
        // Offsets {-1, 0, 1, 2}: distances = {(k-1) - frac} for k=0..3
        int32_t weights_h_fixed[4];
        int32_t weights_w_fixed[4];
        for (int32_t k = 0; k < 4; k++) {
            weights_h_fixed[k] = cubic_weight_fixed(int_to_fixed(k - 1) - frac_h);
            weights_w_fixed[k] = cubic_weight_fixed(int_to_fixed(k - 1) - frac_w);
        }

        // 16 2D weights: outer product → convert to BF16
        uint16_t weights_bf16[16];
        for (int32_t dy = 0; dy < 4; dy++) {
            for (int32_t dx = 0; dx < 4; dx++) {
                int32_t w2d = fixed_mul(weights_h_fixed[dy], weights_w_fixed[dx]);
                weights_bf16[dy * 4 + dx] = signed_fixed_to_bf16(w2d);
            }
        }

        // Compute 16 clamped source page IDs
        const uint32_t batch_offset = batch * input_height * input_width;
        uint32_t src_page_ids[16];
        for (int32_t dy = -1; dy <= 2; dy++) {
            int32_t sy = clamp_coord(floor_h + dy, max_h);
            for (int32_t dx = -1; dx <= 2; dx++) {
                int32_t sx = clamp_coord(floor_w + dx, max_w);
                src_page_ids[(dy + 1) * 4 + (dx + 1)] =
                    batch_offset + static_cast<uint32_t>(sy) * input_width + static_cast<uint32_t>(sx);
            }
        }

        // Push 4 groups × blocks to CBs
        uint32_t block_offset = 0;
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t current_block_bytes = (b == blocks - 1) ? last_block_size_bytes : input_block_size_bytes;

            for (uint32_t g = 0; g < 4; g++) {
                cb_reserve_back(input_cb, 4);
                uint32_t l1_write_addr = get_write_ptr(input_cb);

                for (uint32_t n = 0; n < 4; n++) {
                    uint32_t neighbor_idx = g * 4 + n;
                    uint64_t src_noc_addr = input_tensor_accessor.get_noc_addr(src_page_ids[neighbor_idx]);
                    noc_async_read(src_noc_addr + block_offset, l1_write_addr, current_block_bytes);
                    l1_write_addr += input_block_size_bytes;
                }
                noc_async_read_barrier();

                cb_reserve_back(scalar_cb, 1);
                volatile tt_l1_ptr uint32_t* weight_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scalar_cb));
                uint32_t base = g * 4;
                weight_ptr[0] = static_cast<uint32_t>(weights_bf16[base + 0]) |
                                (static_cast<uint32_t>(weights_bf16[base + 1]) << 16);
                weight_ptr[1] = static_cast<uint32_t>(weights_bf16[base + 2]) |
                                (static_cast<uint32_t>(weights_bf16[base + 3]) << 16);
                cb_push_back(scalar_cb, 1);

                cb_push_back(input_cb, 4);
            }

            block_offset += current_block_bytes;
        }

        pixel_id++;
    }
}
