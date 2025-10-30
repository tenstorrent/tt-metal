// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

#define ALWI inline __attribute__((always_inline))

// Fill given four values into the memory starting at the given address.
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);

    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

// Fixed-point math constants and helpers for Q16.16 format
constexpr int32_t FIXED_POINT_SHIFT = 16;
constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;         // 1.0 in Q16.16
constexpr int32_t FIXED_HALF = 1 << (FIXED_POINT_SHIFT - 1);  // 0.5 in Q16.16

// Extract integer part from fixed-point
ALWI constexpr int32_t fixed_to_int(int32_t fixed) { return fixed >> FIXED_POINT_SHIFT; }

// Extract fractional part from fixed-point (0 to FIXED_ONE)
ALWI constexpr int32_t fixed_frac(int32_t fixed) { return fixed & ((1 << FIXED_POINT_SHIFT) - 1); }

// Multiply two fixed-point numbers
ALWI constexpr int32_t fixed_mul(int32_t a, int32_t b) { return ((int64_t)a * b) >> FIXED_POINT_SHIFT; }

ALWI uint16_t float_to_bfloat16_non_constexpr(float val) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&val);
    return uint16_t(*p >> 16);
}

// Convert fixed-point to bfloat16 for weight values
ALWI uint16_t fixed_to_bfloat16(int32_t fixed) {
    float fval = (float)fixed / FIXED_ONE;
    return float_to_bfloat16_non_constexpr(fval);
}

void kernel_main() {
    // Only runtime argument - which row in the image this core starts processing
    uint32_t start_input_row_in_image_id = get_arg_val<uint32_t>(0);

    // Moved to compile-time arguments
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t in_image_rows_per_core = get_compile_time_arg_val(1);
    constexpr uint32_t scale_h = get_compile_time_arg_val(2);
    constexpr uint32_t scale_w = get_compile_time_arg_val(3);
    constexpr uint32_t in_w = get_compile_time_arg_val(4);
    constexpr uint32_t out_w = get_compile_time_arg_val(5);
    constexpr uint32_t in_h = get_compile_time_arg_val(6);

    // Existing compile-time arguments (shifted indices)
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(9);
    // These are now already in fixed-point format from the host
    constexpr uint32_t scale_h_inv_fixed_u32 = get_compile_time_arg_val(10);
    constexpr uint32_t scale_w_inv_fixed_u32 = get_compile_time_arg_val(11);
    constexpr uint32_t y_starting_coordinate_fixed_u32 = get_compile_time_arg_val(12);
    constexpr uint32_t x_starting_coordinate_fixed_u32 = get_compile_time_arg_val(13);
    constexpr uint32_t is_reader = get_compile_time_arg_val(14);
    constexpr uint32_t blocks = get_compile_time_arg_val(15);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(16);

    constexpr bool src1_is_dram = false;
    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    constexpr uint32_t total_nsticks_to_process = in_w * scale_w;
    // Calculate the number of sticks to process per core by dividing the total number of sticks (in width direction)
    // by 2.
    constexpr uint32_t nsticks_reader =
        (total_nsticks_to_process % 2) ? total_nsticks_to_process / 2 + 1 : total_nsticks_to_process / 2;
    constexpr uint32_t nsticks_writer =
        (total_nsticks_to_process % 2) ? total_nsticks_to_process / 2 : total_nsticks_to_process / 2;
    constexpr uint32_t nsticks_to_process_on_core = is_reader ? nsticks_reader : nsticks_writer;
    // assuming shard begins with a new row. TODO: generalize?
    // Values are already in fixed-point format from host, just cast them
    constexpr int32_t scale_h_inv = static_cast<int32_t>(scale_h_inv_fixed_u32);
    constexpr int32_t scale_w_inv = static_cast<int32_t>(scale_w_inv_fixed_u32);
    constexpr int32_t scale_w_inv_x2 = scale_w_inv << 1;  // scale_w_inv * 2, used frequently
    int32_t y_coordinate,
        x_coordinate;  // x and y coordinate of the output pixel, as if it had existed in the input image
    int32_t x;         // helper variable to avoid out of bound reads in the x direction
    int32_t dx,
        dy;  // distance between the output pixel and its closest pixel with lower coordinates (up and to the left)
    constexpr int32_t y_starting_coordinate_fixed = static_cast<int32_t>(y_starting_coordinate_fixed_u32);
    constexpr int32_t x_starting_coordinate_fixed = static_cast<int32_t>(x_starting_coordinate_fixed_u32);

    y_coordinate = y_starting_coordinate_fixed;

    // If the current core is a writer core, adjust the x_starting_coordinate to start from the correct position.
    constexpr int32_t x_starting_coordinate =
        (!is_reader) ? x_starting_coordinate_fixed + scale_w_inv : x_starting_coordinate_fixed;
    int32_t accumulated_offset = 0;  // offset used to calculate the position of row in batch
                                     // every time we encounter a boundary between images, it is increased by
                                     // in_h (corresponding to height of a single batch) +
                                     // 2 (corresponding to the 2 skipped padding rows)

    noc_async_read_one_packet_set_state(get_noc_addr(l1_read_addr), input_block_size_bytes);
    constexpr uint32_t img2_stick_bytes = in_w * stick_nbytes;
    // DPRINT << "num outer loops: " << in_image_rows_per_core * scale_h << ENDL();
    // DPRINT << "scale_h: " << scale_h << ENDL();
    constexpr uint32_t num_outer_loops = in_image_rows_per_core * scale_h;
    for (uint32_t image_row = 0; image_row < num_outer_loops; ++image_row) {
        x_coordinate = x_starting_coordinate;

        // These variables are for referencing the appropriate input rows, specifically
        // Their coordinates in the halo shard

        uint32_t y1 = fixed_to_int(y_coordinate);
        uint32_t y2 = y1 + 1;

        // These two variables represent the indices of the rows referenced by y1 and y2
        // In the according batch in the input image
        // Value of -1 for either of these corresponds to the top padding row,
        // And value of in_h corresponds to the bottom padding row

        int32_t in_batch_index_y1 = int32_t(y1) - 1 + start_input_row_in_image_id - accumulated_offset;
        int32_t in_batch_index_y2 = int32_t(y2) - 1 + start_input_row_in_image_id - accumulated_offset;

        dy = fixed_frac(y_coordinate);

        // In no circumstance should the padding rows have weights greater than 0.5

        if (in_batch_index_y1 == -1) {
            // This case handles the error on the top border of the image, where the top padding row could be wrongly
            // included

            // Entering this case means that in_batch_index_y1 (the "upper" row) corresponds to a padding row
            // (specifically the top padding row) Reduce the padding row's weight to 0 and put full weight on the row
            // below it
            dy = FIXED_ONE;
        }

        if (in_batch_index_y2 == int(in_h)) {
            // This case handles the error on the bottom border of the image, where the bottom padding could be wrongly
            // included

            // Entering this case means that in_batch_index_y2 (the "lower row") corresponds to a padding
            // row(specifically the bottom padding row)

            if (dy > FIXED_HALF) {  // Due to math behind bilinear upsampling, dy could never be exactly 0.5, so this
                                    // check should be numerically safe

                // In this case, a padding row has weight higher than 0.5.
                // This means we  are actually done with the current batch,
                // and have to skip the next 2 rows (lower padding of current image and upper padding of
                // previous image) The iteration that enters this case outputs the first row of the next batch
                y1 += 2;
                y2 += 2;
                y_coordinate += (2 << FIXED_POINT_SHIFT);  // Add 2.0 in fixed-point
                dy = FIXED_ONE;
                accumulated_offset += 2 + in_h;
            } else {
                // In this case, we handle the rows on the bottom border of the image

                // We should do no skipping, but just set weight to 0 for the padding row (y2)
                dy = 0;
            }
        }
        // DPRINT << "x increment: " << scale_w_inv * 2 << ENDL();
        // DPRINT << "nsticks_to_process_on_core: " << nsticks_to_process_on_core << ENDL();

        // Check if we can use the optimization for pre-computed weights
        // The optimization only works when dx alternates between at most 2 values
        // This happens when scale_w is 1, 2, or 4
        constexpr bool use_precomputed_weights = (scale_w == 1 || scale_w == 2 || scale_w == 4);

        int32_t one_minus_dy = FIXED_ONE - dy;

        // Pre-compute row base addresses (constant for entire row)
        uint32_t y1_base = l1_read_addr + y1 * img2_stick_bytes;
        uint32_t y2_base = l1_read_addr + y2 * img2_stick_bytes;

        // Variables for pre-computed weights (only used when optimization is enabled)
        // These are only used when use_precomputed_weights is true, so we can avoid
        // initializing them otherwise
        [[maybe_unused]] int32_t dx_a, dx_b;
        [[maybe_unused]] bool has_special_first;
        [[maybe_unused]] uint16_t p1_bf16_set_a, p2_bf16_set_a, p3_bf16_set_a, p4_bf16_set_a;
        [[maybe_unused]] uint16_t p1_bf16_set_b, p2_bf16_set_b, p3_bf16_set_b, p4_bf16_set_b;
        [[maybe_unused]] uint16_t p1_bf16_zero, p2_bf16_zero, p3_bf16_zero, p4_bf16_zero;

        if constexpr (use_precomputed_weights) {
            // Pre-calculate bilinear interpolation weights for the two alternating dx values
            // Since dy is constant for this row and dx alternates between two values,
            // we can pre-compute the weights and avoid expensive floating-point operations

            // Pre-compute dx values for the alternating pattern
            // When x increments by scale_w_inv*2, we get a repeating pattern
            // Handle special case when starting x is negative (gets clamped to 0)
            has_special_first = (x_coordinate < 0);

            // Calculate the two alternating dx values based on the actual increment
            // For reader core, after any special first value, dx alternates between two values
            if (has_special_first) {
                // First position is special (dx=0), calculate next two
                int32_t x1 = x_coordinate + (scale_w_inv_x2);  // scale_w_inv * 2
                dx_a = fixed_frac(x1);                         // This will be the first regular dx
                int32_t x2 = x1 + (scale_w_inv_x2);            // x1 + scale_w_inv * 2
                dx_b = fixed_frac(x2);                         // This will be the second regular dx
            } else {
                // No special first, calculate the two alternating values
                int32_t x1 = x_coordinate;
                dx_a = fixed_frac(x1);
                int32_t x2 = x1 + (scale_w_inv_x2);  // x1 + scale_w_inv * 2
                dx_b = fixed_frac(x2);
            }

            // Pre-compute weights for the two alternating dx values
            int32_t one_minus_dx_a = FIXED_ONE - dx_a;
            int32_t one_minus_dx_b = FIXED_ONE - dx_b;

            // Weight set A
            int32_t p1_set_a = fixed_mul(one_minus_dx_a, one_minus_dy);
            int32_t p2_set_a = fixed_mul(dx_a, one_minus_dy);
            int32_t p3_set_a = fixed_mul(one_minus_dx_a, dy);
            int32_t p4_set_a = fixed_mul(dx_a, dy);

            // Weight set B
            int32_t p1_set_b = fixed_mul(one_minus_dx_b, one_minus_dy);
            int32_t p2_set_b = fixed_mul(dx_b, one_minus_dy);
            int32_t p3_set_b = fixed_mul(one_minus_dx_b, dy);
            int32_t p4_set_b = fixed_mul(dx_b, dy);

            // Convert weights to bfloat16 once
            p1_bf16_set_a = fixed_to_bfloat16(p1_set_a);
            p2_bf16_set_a = fixed_to_bfloat16(p2_set_a);
            p3_bf16_set_a = fixed_to_bfloat16(p3_set_a);
            p4_bf16_set_a = fixed_to_bfloat16(p4_set_a);

            p1_bf16_set_b = fixed_to_bfloat16(p1_set_b);
            p2_bf16_set_b = fixed_to_bfloat16(p2_set_b);
            p3_bf16_set_b = fixed_to_bfloat16(p3_set_b);
            p4_bf16_set_b = fixed_to_bfloat16(p4_set_b);

            // Special case weights for dx=0 (only used when x starts negative)
            // Formula: p1=(1-dx)*(1-dy), p2=dx*(1-dy), p3=(1-dx)*dy, p4=dx*dy
            // When dx=0: p1=(1-dy), p2=0, p3=dy, p4=0
            p1_bf16_zero = fixed_to_bfloat16(one_minus_dy);
            p2_bf16_zero = 0;
            p3_bf16_zero = fixed_to_bfloat16(dy);
            p4_bf16_zero = 0;
        }

        for (uint32_t j = 0; j < nsticks_to_process_on_core; j++) {
            // Calculate x position and indices (needed for memory addressing)
            x = x_coordinate < 0 ? 0 : x_coordinate;

            // Select or compute weights based on optimization mode
            uint16_t p1_bf16, p2_bf16, p3_bf16, p4_bf16;

            if constexpr (use_precomputed_weights) {
                // Use pre-computed weights for optimized scale factors
                if (has_special_first && j == 0) {
                    // Special first case where x was negative and clamped to 0
                    p1_bf16 = p1_bf16_zero;
                    p2_bf16 = p2_bf16_zero;
                    p3_bf16 = p3_bf16_zero;
                    p4_bf16 = p4_bf16_zero;
                } else {
                    // Regular alternating pattern
                    bool use_set_a;
                    if (has_special_first) {
                        use_set_a = ((j - 1) % 2) == 0;
                    } else {
                        // No special first, simple alternation
                        use_set_a = (j % 2) == 0;
                    }

                    if (use_set_a) {
                        p1_bf16 = p1_bf16_set_a;
                        p2_bf16 = p2_bf16_set_a;
                        p3_bf16 = p3_bf16_set_a;
                        p4_bf16 = p4_bf16_set_a;
                    } else {
                        p1_bf16 = p1_bf16_set_b;
                        p2_bf16 = p2_bf16_set_b;
                        p3_bf16 = p3_bf16_set_b;
                        p4_bf16 = p4_bf16_set_b;
                    }
                }
            } else {
                // Calculate weights dynamically for other scale factors
                dx = fixed_frac(x);
                int32_t one_minus_dx = FIXED_ONE - dx;
                p1_bf16 = fixed_to_bfloat16(fixed_mul(one_minus_dx, one_minus_dy));
                p2_bf16 = fixed_to_bfloat16(fixed_mul(dx, one_minus_dy));
                p3_bf16 = fixed_to_bfloat16(fixed_mul(one_minus_dx, dy));
                p4_bf16 = fixed_to_bfloat16(fixed_mul(dx, dy));
            }
            uint32_t x1 = fixed_to_int(x);
            uint32_t x2 = (x1 + 1) < (in_w - 1) ? (x1 + 1) : (in_w - 1);

            // Pre-compute column offsets (hoist multiplications out of blocks loop)
            uint32_t x1_offset = x1 * stick_nbytes;
            uint32_t x2_offset = x2 * stick_nbytes;

            uint32_t block_offset = 0;
            for (uint32_t i = 0; i < blocks; i++) {
                cb_reserve_back(out_cb_id, 4);

                uint32_t l1_write_addr = get_write_ptr(out_cb_id);
                uint32_t l1_read_addr_temp = y1_base + x1_offset + block_offset;
                // 1st tile
                noc_async_read_one_packet_with_state<true>(l1_read_addr_temp, l1_write_addr);
                l1_write_addr += input_block_size_bytes;

                // 2nd tile
                l1_read_addr_temp = y1_base + x2_offset + block_offset;
                noc_async_read_one_packet_with_state<true>(l1_read_addr_temp, l1_write_addr);
                l1_write_addr += input_block_size_bytes;

                // 3rd tile
                l1_read_addr_temp = y2_base + x1_offset + block_offset;
                noc_async_read_one_packet_with_state<true>(l1_read_addr_temp, l1_write_addr);
                l1_write_addr += input_block_size_bytes;

                // 4th tile
                l1_read_addr_temp = y2_base + x2_offset + block_offset;
                noc_async_read_one_packet_with_state<true>(l1_read_addr_temp, l1_write_addr);
                l1_write_addr += input_block_size_bytes;

                fill_four_val(get_write_ptr(in_scalar_cb_id), p1_bf16, p2_bf16, p3_bf16, p4_bf16);
                cb_push_back(in_scalar_cb_id, 1);

                // push scaler and data into cb.
                noc_async_read_barrier();
                cb_push_back(out_cb_id, 4);
                block_offset += input_block_size_bytes;
            }
            x_coordinate += (scale_w_inv_x2);  // scale_w_inv * 2 in fixed-point
        }
        y_coordinate += scale_h_inv;
    }
}
