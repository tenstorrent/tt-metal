// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // Retrieve arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_idx = get_arg_val<uint32_t>(1);
    uint32_t end_tile_idx = get_arg_val<uint32_t>(2);
    uint32_t start_padding_tile_idx = get_arg_val<uint32_t>(3);
    uint32_t end_padding_tile_idx = get_arg_val<uint32_t>(4);

    // Compile-time constants
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t element_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(6);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(7);
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(8);
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(9);
    constexpr bool needs_padding = get_compile_time_arg_val(10) == 1;

    // Derived compile-time constants
    constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
    constexpr uint8_t NUM_FACES_H = TILE_HEIGHT / FACE_HEIGHT;
    constexpr uint8_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;

    constexpr uint32_t C_p = tt::data_movement::common::round_up<C, TILE_HEIGHT>();
    constexpr uint32_t H_p = tt::data_movement::common::round_up<H, TILE_HEIGHT>();
    constexpr uint32_t W_p = tt::data_movement::common::round_up<W, TILE_WIDTH>();

    constexpr uint32_t W_t = W_p / TILE_WIDTH;
    constexpr uint32_t H_t = H_p / TILE_HEIGHT;
    constexpr uint32_t C_t = C_p / TILE_HEIGHT;

    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_WIDTH * element_size;

    // Initialize address generator
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const auto input_data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram, TILE_HW> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = input_data_format};

    // Calculate actual data height in the last tile
    constexpr uint32_t H_last_tile = H - (H_t - 1) * TILE_HEIGHT;

    // Calculate real_faces_h
    uint8_t remainder_faces_h = tt::data_movement::common::div_up<H_last_tile, FACE_HEIGHT>();

    uint32_t remainder = H_last_tile % FACE_HEIGHT;
    uint8_t sub_tile_lines_real = (remainder == 0) ? FACE_HEIGHT : static_cast<uint8_t>(remainder);

    // Precompute constants used in inner loops
    const uint32_t face_height_width = FACE_HEIGHT * FACE_WIDTH;
    const uint32_t num_faces_wh = NUM_FACES_W * FACE_WIDTH;

    // Main single loop over all tiles
    for (uint32_t tile_idx = start_tile_idx; tile_idx < end_tile_idx; ++tile_idx) {
        // Compute n, c, h, w from tile_idx
        uint32_t w = tile_idx % W_t;
        uint32_t temp = tile_idx / W_t;

        uint32_t h = temp % H_t;
        temp /= H_t;

        uint32_t c = temp % C;
        uint32_t n = temp / C;

        // Recalculate variables from the original loops
        uint32_t output_ct_index = c / TILE_HEIGHT;
        uint32_t rem = c % TILE_HEIGHT;

        // Calculate the index inside the face_matrix
        uint32_t output_face_h = rem / FACE_HEIGHT;
        uint32_t output_sub_tile_line = rem % FACE_HEIGHT;

        // Precompute offset for the current face_h
        uint32_t face_h_offset = output_face_h * NUM_FACES_W * face_height_width;

        // Calculate the index along the channel dimension for the output tensor
        uint32_t output_h = h * TILE_HEIGHT;

        // Synchronization and read address retrieval
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        // Determine the number of faces in the height dimension
        uint8_t num_faces_h = (h == H_t - 1) ? remainder_faces_h : NUM_FACES_H;

        // Precompute parts of linear_idx that remain constant within the inner loops
        // linear_idx = n * H * C_t * W_t + output_h_face_line * C_t * W_t + output_ct_index * W_t + w
        // We can precompute n * H * C_t * W_t + output_ct_index * W_t + w
        uint32_t base_linear_idx = n * H * C_t * W_t + output_ct_index * W_t + w;

        // Iterate over faces in the height dimension
        for (uint8_t face_h = 0; face_h < num_faces_h; ++face_h) {
            // Compute output_h_face once per face_h
            uint32_t output_h_face = output_h + face_h * FACE_HEIGHT;

            // Precompute the additive factor for output_h_face_line
            uint32_t base_output_h_face_line = output_h_face;

            // Determine the number of sub-tile lines to process
            bool is_last_sub_tile_line = (h == H_t - 1) && (face_h == num_faces_h - 1);
            uint8_t sub_tile_lines = is_last_sub_tile_line ? sub_tile_lines_real : FACE_HEIGHT;

            // Iterate over faces in the width dimension
            for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                // Compute output_w_face once per face_w
                uint32_t output_w_face = w + face_w * FACE_WIDTH;

                // Precompute the offset multiplier for the current face_w
                uint32_t face_w_offset = face_w * face_height_width;

                // Compute the offset
                uint32_t offset = (face_h_offset + face_w_offset + output_sub_tile_line * FACE_WIDTH) * element_size;

                // Iterate over sub-tile lines
                for (uint8_t sub_tile_line = 0; sub_tile_line < sub_tile_lines; ++sub_tile_line) {
                    // Compute the complete output_h_face_line
                    uint32_t output_h_face_line = base_output_h_face_line + sub_tile_line;

                    // Compute the linear index
                    uint32_t linear_idx = base_linear_idx + output_h_face_line * C_t * W_t;

                    // Compute the write address
                    uint64_t write_noc_base_addr = get_noc_addr(linear_idx, s, offset);

                    // Perform asynchronous write
                    noc_async_write(l1_read_addr, write_noc_base_addr, SUBTILE_LINE_BYTES);

                    // Increment the read address
                    l1_read_addr += SUBTILE_LINE_BYTES;
                }

                // Skip padding if not all lines are real
                if (is_last_sub_tile_line) {
                    l1_read_addr += (FACE_HEIGHT - sub_tile_lines) * SUBTILE_LINE_BYTES;
                }
            }
        }

        // Ensure all asynchronous writes are completed before proceeding
        noc_async_write_barrier();

        // Remove the processed tile from the front of the buffer
        cb_pop_front(cb_id_out0, 1);
    }

    // add padding
    if constexpr (needs_padding) {
        cb_wait_front(tt::CBIndex::c_1, 1);

        uint32_t l1_read_ptr = get_read_ptr(tt::CBIndex::c_1);

        constexpr uint32_t c_t = C_t - 1;
        constexpr uint8_t C_in_tile = C % TILE_HEIGHT;
        constexpr uint8_t face_c_start = C_in_tile / FACE_HEIGHT;

        for (uint32_t tile_idx = start_padding_tile_idx; tile_idx < end_padding_tile_idx; ++tile_idx) {
            // Map tile_idx to (n, h, w_t)
            uint32_t n = tile_idx / (H * W_t);
            uint32_t remainder1 = tile_idx % (H * W_t);
            uint32_t h = remainder1 / W_t;
            uint32_t w_t = remainder1 % W_t;

            // Calculate linear_idx of padded tile inside output tensor buffer
            uint32_t linear_idx = n * H * C_t * W_t + h * C_t * W_t + c_t * W_t + w_t;

            for (uint8_t face_c = face_c_start; face_c < NUM_FACES_H; ++face_c) {
                // Offset to the start of the current face along the channel dimension/height of the tile
                uint32_t face_c_offset = face_c * NUM_FACES_W * face_height_width;

                // Sub-tile/face line where our padded data starts
                uint8_t sub_tile_line_start = face_c == face_c_start ? C_in_tile % FACE_HEIGHT : 0;

                for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                    // Offset to the start of the current face along the width of the tile
                    uint32_t face_w_offset = face_w * face_height_width;
                    for (uint8_t sub_tile_line = sub_tile_line_start; sub_tile_line < FACE_HEIGHT; ++sub_tile_line) {
                        // offset to the start of the current sub-tile line
                        uint32_t offset = (face_c_offset + face_w_offset + sub_tile_line * FACE_WIDTH) * element_size;

                        // Compute the write address
                        uint64_t write_noc_base_addr = get_noc_addr(linear_idx, s, offset);

                        // Perform asynchronous write
                        noc_async_write(l1_read_ptr, write_noc_base_addr, SUBTILE_LINE_BYTES);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(tt::CBIndex::c_1, 1);
    }
}
