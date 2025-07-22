// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // X = output width
    // Y = output height
    // input shape = (..., H, W)
    // output shape = (..., Y, X)

    /**
     * The reader kernel reads in a XW block, and then compute kernel transposes the XW block into a WX block
     * X is our output width dimension, so the X values are contiguous in the buffer
     * As a result we can write them out into chunks
     * However, since our output has subtiles/faces, where each face line is 16 elements, we have bubbles every 16
     * elements Thus we can only write out 16 elements at a time This kernel takes in the transpose XW block (now WX
     * where X is the contiguous dimension) and writes out the subtile/face lines along the X to their final, permuted
     * positions in the output tensor. If the Y dimension is not a multiple of TILE_HEIGHT, we pad the last set of tiles
     * along the Y dimension with the pad value until it is
     */

    //--------------------------------------------------------------------------
    // 1) Compile-time Arguments
    //--------------------------------------------------------------------------

    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t RANK = get_compile_time_arg_val(1);
    // constexpr uint32_t input_cb_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t element_size = get_compile_time_arg_val(3);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(4);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(5);
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(6);
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(7);
    constexpr uint32_t x_dim_in_input = get_compile_time_arg_val(8);
    constexpr uint32_t X = get_compile_time_arg_val(9);
    constexpr uint32_t W = get_compile_time_arg_val(10);
    constexpr uint32_t Y = get_compile_time_arg_val(11);
    constexpr uint32_t X_p = get_compile_time_arg_val(12);
    constexpr uint32_t W_p = get_compile_time_arg_val(13);
    constexpr uint32_t rows_per_x = get_compile_time_arg_val(14);
    // 15 is Y_t, see below
    constexpr uint32_t W_t = get_compile_time_arg_val(16);
    constexpr uint32_t final_tile_real_x = get_compile_time_arg_val(17);
    constexpr uint32_t final_tile_real_faces_x = get_compile_time_arg_val(18);
    constexpr uint32_t xw_blocks = get_compile_time_arg_val(19);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(20);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(21);
    constexpr bool needs_y_padding = (bool)get_compile_time_arg_val(22);
    constexpr uint32_t permuted_w_dim = get_compile_time_arg_val(23);

    //--------------------------------------------------------------------------
    // 2) Derived Constants (all constexpr)
    //--------------------------------------------------------------------------
    constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t FACE_HW = FACE_HEIGHT * FACE_WIDTH;
    constexpr uint32_t FACE_HW_BYTES = FACE_HW * element_size;
    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_WIDTH * element_size;
    constexpr uint32_t TILE_LINE_BYTES = TILE_WIDTH * element_size;
    constexpr uint32_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;
    constexpr uint32_t NUM_FACES_H = TILE_HEIGHT / FACE_HEIGHT;
    constexpr uint32_t x_block_size = TILE_HEIGHT;
    constexpr uint32_t w_block_size = TILE_WIDTH;
    constexpr uint32_t FACE_H_STRIDE_BYTES = NUM_FACES_W * FACE_HW_BYTES;

    constexpr uint32_t tile_bytes = TILE_HW * element_size;
    constexpr uint32_t w_dim = RANK - 1;

    // For output height, tile-based:
    constexpr uint32_t H_p = TILE_HEIGHT * ((Y + TILE_HEIGHT - 1) / TILE_HEIGHT);
    constexpr uint32_t Y_t = H_p / TILE_HEIGHT;
    // For X dimension:
    constexpr uint32_t X_t = X_p / TILE_HEIGHT;

    constexpr auto cb_out = tt::CBIndex::c_2;

    //--------------------------------------------------------------------------
    // 3) Runtime Arguments
    //--------------------------------------------------------------------------
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block = get_arg_val<uint32_t>(1);
    uint32_t end_block = get_arg_val<uint32_t>(2);
    uint32_t start_padding_tile_idx = get_arg_val<uint32_t>(3);
    uint32_t end_padding_tile_idx = get_arg_val<uint32_t>(4);

    constexpr uint32_t array_start_offset = 5;
    uint32_t input_shape[RANK], dims[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + array_start_offset);
        dims[i] = get_arg_val<uint32_t>(i + RANK + array_start_offset);
    }

    // Build the permuted output shape
    uint32_t output_shape[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        output_shape[i] = input_shape[dims[i]];
    }

    //--------------------------------------------------------------------------
    // 4) Build padded/tiled shapes
    //--------------------------------------------------------------------------
    uint32_t output_tiled_shape[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        if (i < RANK - 2) {
            output_tiled_shape[i] = output_shape[i];
        } else if (i == RANK - 2) {
            output_tiled_shape[i] = Y_t;
        } else {
            output_tiled_shape[i] = X_t;  // i == RANK - 1
        }
    }

    //--------------------------------------------------------------------------
    // 5) Row strides in the padded shape
    //--------------------------------------------------------------------------
    uint32_t dest_tiled_strides[RANK];
    dest_tiled_strides[RANK - 1] = 1;
    for (int i = RANK - 2; i >= 0; i--) {
        dest_tiled_strides[i] = dest_tiled_strides[i + 1] * output_tiled_shape[i + 1];
    }

    // The stride for stepping along dimension `permuted_w_dim` in the final output
    uint32_t W_stride_tile = dest_tiled_strides[permuted_w_dim];

    constexpr auto data_format = get_dataformat(tt::CBIndex::c_0);
    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t idxs[RANK];
    idxs[RANK - 1] = 0;
    uint32_t dest_multi_idx[RANK];

    // If the new X dimension can have partial faces in the last tile:
    uint32_t real_faces_x = 0;
    if constexpr (NUM_FACES_W == final_tile_real_faces_x) {
        real_faces_x = NUM_FACES_W;
    }

    //--------------------------------------------------------------------------
    // 6) Main Loop: writing each block
    //--------------------------------------------------------------------------
    for (uint32_t block = start_block; block < end_block; block++) {
        // Decompose block => w_block, x_block, xw_block
        uint32_t rem = block;
        const uint32_t w_block = rem % w_blocks;
        rem /= w_blocks;

        const uint32_t x_block = rem % x_blocks;
        rem /= x_blocks;

        uint32_t xw_block = rem % rows_per_x;
        uint32_t remainder = xw_block;

        // W range
        uint32_t w_start = w_block * w_block_size;
        uint32_t w_end =
            (w_start + w_block_size < input_shape[RANK - 1]) ? (w_start + w_block_size) : input_shape[RANK - 1];

        // Fill idxs except for x_dim_in_input
        for (int32_t d = RANK - 2; d >= 0; --d) {
            if (d == (int32_t)x_dim_in_input) {
                idxs[d] = 0;
                continue;
            }
            idxs[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }
        idxs[RANK - 1] = w_start;

        // Build dest_multi_idx from dims[]
        for (uint32_t d = 0; d < RANK; d++) {
            dest_multi_idx[d] = idxs[dims[d]];
        }

        // If permuted_w_dim != RANK-2, we compute sub-tile offsets once per block
        // else we recalc them inside the W loop below.
        uint32_t y = 0;
        uint8_t output_sub_tile_line = 0;
        uint8_t output_face_h = 0;
        uint16_t base_face_line_offset_bytes = 0;

        if constexpr (permuted_w_dim != (RANK - 2)) {
            y = dest_multi_idx[RANK - 2];
            output_sub_tile_line = y % FACE_HEIGHT;
            output_face_h = (y % TILE_HEIGHT) / FACE_HEIGHT;
            base_face_line_offset_bytes =
                static_cast<uint16_t>(output_face_h * FACE_H_STRIDE_BYTES + output_sub_tile_line * SUBTILE_LINE_BYTES);
        }

        // The tile index for the RANK-2 dimension
        dest_multi_idx[RANK - 2] /= TILE_HEIGHT;
        // The tile index for the last dimension => x_block
        dest_multi_idx[RANK - 1] = x_block;

        // Flatten => base_tile_offset
        uint32_t base_tile_offset = 0;
        for (uint32_t i = 0; i < RANK; i++) {
            if constexpr (permuted_w_dim != (RANK - 2)) {
                if (i == permuted_w_dim) {
                    continue;
                }
            }
            base_tile_offset += dest_multi_idx[i] * dest_tiled_strides[i];
        }

        if constexpr (NUM_FACES_W != final_tile_real_faces_x) {
            real_faces_x = (x_block == (X_t - 1)) ? final_tile_real_faces_x : NUM_FACES_W;
        }

        // Wait for 1 tile from cb_out
        cb_wait_front(cb_out, 1);
        uint32_t transposed_buffer_read_addr = get_read_ptr(cb_out);

        // ---------------------------------------------------------------------
        // 6.1) Write out each W in [w_start..w_end)
        // ---------------------------------------------------------------------
        {
            uint32_t page_offset = 0;  // increments by TILE_LINE_BYTES

            for (uint32_t w = w_start; w < w_end; ++w) {
                uint16_t local_face_offset_bytes = base_face_line_offset_bytes;
                uint32_t tile = 0;

                if constexpr (permuted_w_dim != (RANK - 2)) {
                    tile = base_tile_offset + (w * W_stride_tile);
                } else {
                    // Recompute sub-tile offsets for each W
                    output_sub_tile_line = (w % FACE_HEIGHT);
                    output_face_h = (w % TILE_HEIGHT) / FACE_HEIGHT;
                    local_face_offset_bytes = static_cast<uint16_t>(
                        (output_face_h * FACE_H_STRIDE_BYTES) + (output_sub_tile_line * SUBTILE_LINE_BYTES));
                    tile = base_tile_offset;
                }

                uint64_t dest_noc_addr = get_noc_addr(tile, s, local_face_offset_bytes);
                uint32_t l1_row_base = transposed_buffer_read_addr + page_offset;

                // Write each face
                uint16_t x_offset = 0;
                uint16_t cb_x_offset = 0;
                for (uint8_t i = 0; i < real_faces_x; i++) {
                    noc_async_write(l1_row_base + cb_x_offset, dest_noc_addr + x_offset, SUBTILE_LINE_BYTES);

                    x_offset += FACE_HW_BYTES;
                    cb_x_offset += SUBTILE_LINE_BYTES;
                }

                page_offset += TILE_LINE_BYTES;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }

    //--------------------------------------------------------------------------
    // 7) Y Padding if needed
    //--------------------------------------------------------------------------
    if constexpr (needs_y_padding) {
        // Some sites used const or runtime logic; here we do everything at compile time:
        constexpr uint32_t y_t = Y_t - 1;
        constexpr uint8_t Y_in_tile = (Y % TILE_HEIGHT);
        constexpr uint8_t face_y_start = (Y_in_tile / FACE_HEIGHT);

        // We'll reuse 'dest_multi_idx' for tile indexing
        dest_multi_idx[RANK - 2] = y_t;  // fix the tile dimension in the RANK-2 dimension

        cb_wait_front(tt::CBIndex::c_3, 1);
        uint32_t l1_read_ptr = get_read_ptr(tt::CBIndex::c_3);

        for (uint32_t tile_idx = start_padding_tile_idx; tile_idx < end_padding_tile_idx; ++tile_idx) {
            // Unflatten 'tile_idx' => dest_multi_idx
            size_t remaining = tile_idx;
            for (uint32_t i = 0; i < RANK; ++i) {
                size_t dim = RANK - 1 - i;
                if (dim == (RANK - 2)) {
                    continue;
                }
                dest_multi_idx[dim] = (remaining % output_tiled_shape[dim]);
                remaining /= output_tiled_shape[dim];
            }

            // Flatten => linear_idx
            uint32_t linear_idx = 0;
            for (uint32_t i = 0; i < RANK; ++i) {
                linear_idx += dest_tiled_strides[i] * dest_multi_idx[i];
            }

            // Write out padding lines
            for (uint8_t face_y = face_y_start; face_y < NUM_FACES_H; ++face_y) {
                uint16_t face_y_offset = face_y * NUM_FACES_W * FACE_HW;
                uint8_t sub_tile_line_start = (face_y == face_y_start) ? (Y_in_tile % FACE_HEIGHT) : 0;

                for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                    uint16_t face_offset = face_y_offset + (face_w * FACE_HW);

                    for (uint8_t sub_tile_line = sub_tile_line_start; sub_tile_line < FACE_HEIGHT; ++sub_tile_line) {
                        uint16_t offset =
                            static_cast<uint16_t>((face_offset + (sub_tile_line * FACE_WIDTH)) * element_size);
                        uint64_t write_noc_base_addr = get_noc_addr(linear_idx, s, offset);

                        noc_async_write(l1_read_ptr, write_noc_base_addr, SUBTILE_LINE_BYTES);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(tt::CBIndex::c_3, 1);
    }
}
