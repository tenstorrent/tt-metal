// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "fixed_point_arithmetic.h"

// Fill given four values into the memory starting at the given address.
// Used to fill the bilinear weights for reduction into L1 memory.
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);

    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

void kernel_main() {
    // Only runtime argument - which row in the input image this core starts processing
    uint32_t start_input_row_in_image_id = get_arg_val<uint32_t>(0);

    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t in_image_rows_per_core = get_compile_time_arg_val(1);
    constexpr uint32_t scale_h = get_compile_time_arg_val(2);
    constexpr uint32_t scale_w = get_compile_time_arg_val(3);
    constexpr uint32_t in_w = get_compile_time_arg_val(4);
    constexpr uint32_t out_w = get_compile_time_arg_val(5);
    constexpr uint32_t in_h = get_compile_time_arg_val(6);

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

    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    constexpr uint32_t number_output_sticks_per_input_row = in_w * scale_w;
    // Calculate the number of output sticks per row to process per core by dividing the total number of output sticks
    // per row (in width direction) by 2.
    constexpr uint32_t number_output_sticks_per_row_reader_core = (number_output_sticks_per_input_row % 2)
                                                                      ? number_output_sticks_per_input_row / 2 + 1
                                                                      : number_output_sticks_per_input_row / 2;
    constexpr uint32_t number_output_sticks_per_row_writer_core = (number_output_sticks_per_input_row % 2)
                                                                      ? number_output_sticks_per_input_row / 2
                                                                      : number_output_sticks_per_input_row / 2;
    constexpr uint32_t number_output_sticks_per_row_on_core =
        is_reader ? number_output_sticks_per_row_reader_core : number_output_sticks_per_row_writer_core;
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

    // If the current core is a writer core (passed as argument from the factory), adjust the x_starting_coordinate to
    // start from the correct position.
    constexpr int32_t x_starting_coordinate =
        (!is_reader) ? x_starting_coordinate_fixed + scale_w_inv : x_starting_coordinate_fixed;
    int32_t accumulated_offset = 0;  // offset used to calculate the position of row in batch
                                     // every time we encounter a boundary between images, it is increased by
                                     // in_h (corresponding to the height of a single batch) +
                                     // 2 (corresponding to the 2 skipped padding rows)

    constexpr uint32_t img2_stick_bytes = in_w * stick_nbytes;
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
        [[maybe_unused]] int32_t dx_even, dx_odd;
        [[maybe_unused]] bool has_special_first;
        [[maybe_unused]] uint16_t p1_bf16_even, p2_bf16_even, p3_bf16_even, p4_bf16_even;
        [[maybe_unused]] uint16_t p1_bf16_odd, p2_bf16_odd, p3_bf16_odd, p4_bf16_odd;
        [[maybe_unused]] uint16_t p1_bf16_zero, p2_bf16_zero, p3_bf16_zero, p4_bf16_zero;

        if constexpr (use_precomputed_weights) {
            // Pre-calculate bilinear interpolation weights for the even/odd alternating dx values
            // Since dy is constant for this row and dx alternates between even and odd positions,
            // we can pre-compute the weights and avoid expensive floating-point operations

            // Pre-compute dx values for the even/odd alternating pattern
            // When x increments by scale_w_inv*2, we get a repeating even/odd pattern
            // Handle special case when starting x is negative (gets clamped to 0)
            has_special_first = (x_coordinate < 0);

            // Calculate the even/odd alternating dx values based on the actual increment
            // For reader core, after any special first value, dx alternates between even and odd positions
            if (has_special_first) {
                // First position is special (dx=0), calculate next even and odd
                int32_t x1 = x_coordinate + (scale_w_inv_x2);  // scale_w_inv * 2
                dx_even = fixed_frac(x1);                      // This will be the even position dx
                int32_t x2 = x1 + (scale_w_inv_x2);            // x1 + scale_w_inv * 2
                dx_odd = fixed_frac(x2);                       // This will be the odd position dx
            } else {
                // No special first, calculate the even and odd alternating values
                int32_t x1 = x_coordinate;
                dx_even = fixed_frac(x1);
                int32_t x2 = x1 + (scale_w_inv_x2);  // x1 + scale_w_inv * 2
                dx_odd = fixed_frac(x2);
            }

            // Pre-compute weights for the even/odd alternating dx values
            int32_t one_minus_dx_even = FIXED_ONE - dx_even;
            int32_t one_minus_dx_odd = FIXED_ONE - dx_odd;

            // Even position weights
            int32_t p1_even = fixed_mul(one_minus_dx_even, one_minus_dy);
            int32_t p2_even = fixed_mul(dx_even, one_minus_dy);
            int32_t p3_even = fixed_mul(one_minus_dx_even, dy);
            int32_t p4_even = fixed_mul(dx_even, dy);

            // Odd position weights
            int32_t p1_odd = fixed_mul(one_minus_dx_odd, one_minus_dy);
            int32_t p2_odd = fixed_mul(dx_odd, one_minus_dy);
            int32_t p3_odd = fixed_mul(one_minus_dx_odd, dy);
            int32_t p4_odd = fixed_mul(dx_odd, dy);

            // Convert weights to bfloat16 once
            p1_bf16_even = fixed_to_bfloat16(p1_even);
            p2_bf16_even = fixed_to_bfloat16(p2_even);
            p3_bf16_even = fixed_to_bfloat16(p3_even);
            p4_bf16_even = fixed_to_bfloat16(p4_even);

            p1_bf16_odd = fixed_to_bfloat16(p1_odd);
            p2_bf16_odd = fixed_to_bfloat16(p2_odd);
            p3_bf16_odd = fixed_to_bfloat16(p3_odd);
            p4_bf16_odd = fixed_to_bfloat16(p4_odd);

            // Special case weights for dx=0 (only used when x starts negative)
            // Formula: p1=(1-dx)*(1-dy), p2=dx*(1-dy), p3=(1-dx)*dy, p4=dx*dy
            // When dx=0: p1=(1-dy), p2=0, p3=dy, p4=0
            p1_bf16_zero = fixed_to_bfloat16(one_minus_dy);
            p2_bf16_zero = 0;
            p3_bf16_zero = fixed_to_bfloat16(dy);
            p4_bf16_zero = 0;
        }

        for (uint32_t j = 0; j < number_output_sticks_per_row_on_core; j++) {
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
                    // Regular even/odd alternating pattern
                    bool use_even_weights;
                    if (has_special_first) {
                        use_even_weights = ((j - 1) % 2) == 0;
                    } else {
                        // No special first, simple even/odd alternation
                        use_even_weights = (j % 2) == 0;
                    }

                    if (use_even_weights) {
                        p1_bf16 = p1_bf16_even;
                        p2_bf16 = p2_bf16_even;
                        p3_bf16 = p3_bf16_even;
                        p4_bf16 = p4_bf16_even;
                    } else {
                        p1_bf16 = p1_bf16_odd;
                        p2_bf16 = p2_bf16_odd;
                        p3_bf16 = p3_bf16_odd;
                        p4_bf16 = p4_bf16_odd;
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
                // 1st stick
                uint64_t src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 2nd stick
                l1_read_addr_temp = y1_base + x2_offset + block_offset;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 3rd stick
                l1_read_addr_temp = y2_base + x1_offset + block_offset;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 4th stick
                l1_read_addr_temp = y2_base + x2_offset + block_offset;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
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
