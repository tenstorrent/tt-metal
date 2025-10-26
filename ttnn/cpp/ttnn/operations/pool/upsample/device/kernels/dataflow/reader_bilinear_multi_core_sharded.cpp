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

ALWI float uint32_to_float(uint32_t f) {
    float ret;
    std::memcpy(&ret, &f, sizeof(float));
    return ret;
}

void kernel_main() {
    uint32_t stick_nbytes = get_arg_val<uint32_t>(0);
    uint32_t in_image_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t scale_h = get_arg_val<uint32_t>(2);
    uint32_t scale_w = get_arg_val<uint32_t>(3);
    uint32_t in_w = get_arg_val<uint32_t>(4);
    uint32_t out_w = get_arg_val<uint32_t>(5);
    uint32_t start_input_row_in_image_id = get_arg_val<uint32_t>(6);
    uint32_t in_h = get_arg_val<uint32_t>(7);

    constexpr bool src1_is_dram = false;
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(2);
    // constexpr uint32_t is_reader = get_compile_time_arg_val(2);
    constexpr uint32_t scale_h_inv_comp = get_compile_time_arg_val(3);
    constexpr uint32_t scale_w_inv_comp = get_compile_time_arg_val(4);
    constexpr uint32_t y_starting_coordinate_u32 = get_compile_time_arg_val(5);
    constexpr uint32_t x_starting_coordinate_u32 = get_compile_time_arg_val(6);
    constexpr uint32_t is_reader = get_compile_time_arg_val(7);
    constexpr uint32_t blocks = get_compile_time_arg_val(8);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(9);
    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    uint32_t total_nsticks_to_process = in_w * scale_w;
    // Calculate the number of sticks to process per core by dividing the total number of sticks (in width direction)
    // by 2.
    uint32_t nsticks_to_process_on_core =
        (total_nsticks_to_process % 2) ? total_nsticks_to_process / 2 + 1 : total_nsticks_to_process / 2;
    // assuming shard begins with a new row. TODO: generalize?
    float scale_h_inv = uint32_to_float(scale_h_inv_comp);
    float scale_w_inv = uint32_to_float(scale_w_inv_comp);
    float y_coordinate,
        x_coordinate;  // x and y coordinate of the output pixel, as if it had existed in the input image
    float x;           // helper variable to avoid out of bound reads in the x direction
    float dx,
        dy;  // distance between the output pixel and its closest pixel with lower coordinates (up and to the left)
    y_coordinate = uint32_to_float(y_starting_coordinate_u32);
    float x_starting_coordinate = uint32_to_float(x_starting_coordinate_u32);

    // If the current core is a writer core, adjust the x_starting_coordinate to start from the correct position.

    if (!is_reader) {
        x_starting_coordinate += scale_w_inv;
        // If the total number of sticks is odd, process one less stick.
        nsticks_to_process_on_core =
            (total_nsticks_to_process % 2) ? nsticks_to_process_on_core - 1 : nsticks_to_process_on_core;
    }
    int32_t accumulated_offset = 0;  // offset used to calculate the position of row in batch
                                     // every time we encounter a boundary between images, it is increased by
                                     // in_h (corresponding to height of a single batch) +
                                     // 2 (corresponding to the 2 skipped padding rows)

    // DPRINT << "num outer loops: " << in_image_rows_per_core * scale_h << ENDL();
    // DPRINT << "scale_h: " << scale_h << ENDL();
    for (uint32_t image_row = 0; image_row < in_image_rows_per_core * scale_h; ++image_row) {
        x_coordinate = x_starting_coordinate;

        // These variables are for referencing the appropriate input rows, specifically
        // Their coordinates in the halo shard

        uint32_t y1 = int(y_coordinate);
        uint32_t y2 = y1 + 1;

        // These two variables represent the indices of the rows referenced by y1 and y2
        // In the according batch in the input image
        // Value of -1 for either of these corresponds to the top padding row,
        // And value of in_h corresponds to the bottom padding row

        int32_t in_batch_index_y1 = int32_t(y1) - 1 + start_input_row_in_image_id - accumulated_offset;
        int32_t in_batch_index_y2 = int32_t(y2) - 1 + start_input_row_in_image_id - accumulated_offset;

        dy = y_coordinate - y1;

        // In no circumstance should the padding rows have weights greater than 0.5

        if (in_batch_index_y1 == -1) {
            // This case handles the error on the top border of the image, where the top padding row could be wrongly
            // included

            // Entering this case means that in_batch_index_y1 (the "upper" row) corresponds to a padding row
            // (specifically the top padding row) Reduce the padding row's weight to 0 and put full weight on the row
            // below it
            dy = 1;
        }

        if (in_batch_index_y2 == int(in_h)) {
            // This case handles the error on the bottom border of the image, where the bottom padding could be wrongly
            // included

            // Entering this case means that in_batch_index_y2 (the "lower row") corresponds to a padding
            // row(specifically the bottom padding row)

            if (dy > 0.5) {  // Due to math behind bilinear upsampling, dy could never be exactly 0.5, so this check
                             // should be numerically safe

                // In this case, a padding row has weight higher than 0.5.
                // This means we  are actually done with the current batch,
                // and have to skip the next 2 rows (lower padding of current image and upper padding of
                // previous image) The iteration that enters this case outputs the first row of the next batch
                y1 += 2;
                y2 += 2;
                y_coordinate += 2;
                dy = 1;
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
        bool use_precomputed_weights = (scale_w == 1 || scale_w == 2 || scale_w == 4);

        float one_minus_dy = 1 - dy;

        // Variables for pre-computed weights (only used when optimization is enabled)
        float dx_a = 0, dx_b = 0;
        bool has_special_first = false;
        uint16_t p1_bf16_set_a = 0, p2_bf16_set_a = 0, p3_bf16_set_a = 0, p4_bf16_set_a = 0;
        uint16_t p1_bf16_set_b = 0, p2_bf16_set_b = 0, p3_bf16_set_b = 0, p4_bf16_set_b = 0;
        uint16_t p1_bf16_zero = 0, p2_bf16_zero = 0, p3_bf16_zero = 0, p4_bf16_zero = 0;

        if (use_precomputed_weights) {
            // Pre-calculate bilinear interpolation weights for the two alternating dx values
            // Since dy is constant for this row and dx alternates between two values,
            // we can pre-compute the weights and avoid expensive floating-point operations

            // Pre-compute dx values for the alternating pattern
            // When x increments by scale_w_inv*2, we get a repeating pattern
            // Handle special case when starting x is negative (gets clamped to 0)
            float x_temp = x_coordinate;
            has_special_first = (x_temp < 0);

            // Calculate the two alternating dx values based on the actual increment
            // For reader core, after any special first value, dx alternates between two values
            if (has_special_first) {
                // First position is special (dx=0), calculate next two
                float x1 = x_coordinate + scale_w_inv * 2;
                dx_a = x1 - int(x1);  // This will be the first regular dx
                float x2 = x1 + scale_w_inv * 2;
                dx_b = x2 - int(x2);  // This will be the second regular dx
            } else {
                // No special first, calculate the two alternating values
                float x1 = x_coordinate;
                dx_a = x1 - int(x1);
                float x2 = x1 + scale_w_inv * 2;
                dx_b = x2 - int(x2);
            }

            // Pre-compute weights for the two alternating dx values
            float one_minus_dx_a = 1 - dx_a;
            float one_minus_dx_b = 1 - dx_b;

            // Weight set A
            float p1_set_a = one_minus_dx_a * one_minus_dy;
            float p2_set_a = dx_a * one_minus_dy;
            float p3_set_a = one_minus_dx_a * dy;
            float p4_set_a = dx_a * dy;

            // Weight set B
            float p1_set_b = one_minus_dx_b * one_minus_dy;
            float p2_set_b = dx_b * one_minus_dy;
            float p3_set_b = one_minus_dx_b * dy;
            float p4_set_b = dx_b * dy;

            // Convert weights to bfloat16 once
            p1_bf16_set_a = float_to_bfloat16(p1_set_a);
            p2_bf16_set_a = float_to_bfloat16(p2_set_a);
            p3_bf16_set_a = float_to_bfloat16(p3_set_a);
            p4_bf16_set_a = float_to_bfloat16(p4_set_a);

            p1_bf16_set_b = float_to_bfloat16(p1_set_b);
            p2_bf16_set_b = float_to_bfloat16(p2_set_b);
            p3_bf16_set_b = float_to_bfloat16(p3_set_b);
            p4_bf16_set_b = float_to_bfloat16(p4_set_b);

            // Special case weights for dx=0 (only used when x starts negative)
            // Formula: p1=(1-dx)*(1-dy), p2=dx*(1-dy), p3=(1-dx)*dy, p4=dx*dy
            // When dx=0: p1=(1-dy), p2=0, p3=dy, p4=0
            p1_bf16_zero = float_to_bfloat16(one_minus_dy);
            p2_bf16_zero = 0;
            p3_bf16_zero = float_to_bfloat16(dy);
            p4_bf16_zero = 0;
        }

        for (uint32_t j = 0; j < nsticks_to_process_on_core; j++) {
            DeviceZoneScopedN("XLoop");

            // Calculate x position and indices (needed for memory addressing)
            x = x_coordinate < 0 ? 0 : x_coordinate;

            // Select or compute weights based on optimization mode
            uint16_t p1_bf16, p2_bf16, p3_bf16, p4_bf16;

            if (use_precomputed_weights) {
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
                dx = x - int(x);
                p1_bf16 = float_to_bfloat16((1 - dx) * (1 - dy));
                p2_bf16 = float_to_bfloat16(dx * (1 - dy));
                p3_bf16 = float_to_bfloat16((1 - dx) * dy);
                p4_bf16 = float_to_bfloat16(dx * dy);
            }
            uint32_t x1 = int(x);
            uint32_t x2 = (x1 + 1) < (in_w - 1) ? (x1 + 1) : (in_w - 1);

            // Debug output - showing which weights are being used
            // DPRINT << "x_coordinate: " << x_coordinate << ", y_coordinate: " << y_coordinate << ENDL();
            // if (has_special_first && j == 0) {
            //     DPRINT << "dx: 0, dy: " << dy << ENDL();
            //     DPRINT << "p1: " << one_minus_dy << ", p2: 0, p3: " << dy << ", p4: 0" << ENDL();
            // } else {
            //     bool is_set_a = (has_special_first ? ((j - 1) % 2) == 0 : (j % 2) == 0);
            //     float dx_debug = is_set_a ? dx_a : dx_b;
            //     DPRINT << "dx: " << dx_debug << ", dy: " << dy << ENDL();
            //     DPRINT << "p1: " << (is_set_a ? p1_set_a : p1_set_b)
            //            << ", p2: " << (is_set_a ? p2_set_a : p2_set_b)
            //            << ", p3: " << (is_set_a ? p3_set_a : p3_set_b)
            //            << ", p4: " << (is_set_a ? p4_set_a : p4_set_b) << ENDL();
            // }

            for (uint32_t i = 0; i < blocks; i++) {
                cb_reserve_back(out_cb_id, 4);
                cb_reserve_back(in_scalar_cb_id, 1);

                fill_four_val(get_write_ptr(in_scalar_cb_id), p1_bf16, p2_bf16, p3_bf16, p4_bf16);

                uint32_t l1_write_addr = get_write_ptr(out_cb_id);
                uint32_t l1_read_addr_temp =
                    l1_read_addr + x1 * stick_nbytes + y1 * in_w * stick_nbytes + i * input_block_size_bytes;
                // 1st tile
                uint64_t src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 2nd tile
                l1_read_addr_temp =
                    l1_read_addr + y1 * in_w * stick_nbytes + x2 * stick_nbytes + i * input_block_size_bytes;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 3rd tile
                l1_read_addr_temp =
                    l1_read_addr + y2 * in_w * stick_nbytes + x1 * stick_nbytes + i * input_block_size_bytes;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // 4th tile
                l1_read_addr_temp =
                    l1_read_addr + y2 * in_w * stick_nbytes + x2 * stick_nbytes + i * input_block_size_bytes;
                src_noc_addr = get_noc_addr(l1_read_addr_temp);
                noc_async_read(src_noc_addr, l1_write_addr, input_block_size_bytes);
                l1_write_addr += input_block_size_bytes;

                // push scaler and data into cb.
                noc_async_read_barrier();
                cb_push_back(out_cb_id, 4);
                cb_push_back(in_scalar_cb_id, 1);
            }
            x_coordinate += scale_w_inv * 2;
        }
        y_coordinate += scale_h_inv;
    }
}
