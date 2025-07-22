// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

// ------------------------------------------------------------------
// 1) unflatten_index<RANK>:
//    Unflatten 'flat_idx' in row-major order for a shape[] of length RANK.
//    shape[d] is also uint32_t. We store the result into out_multi_idx[].
template <uint32_t RANK>
inline void unflatten_index(uint32_t flat_idx, const uint32_t (&shape)[RANK], uint32_t (&out_multi_idx)[RANK]) {
    // Process from last dimension to first, in row-major unflattening.
    for (int d = RANK - 1; d >= 0; d--) {
        uint32_t dim_size = shape[d];
        out_multi_idx[d] = flat_idx % dim_size;
        flat_idx /= dim_size;
    }
}

// ------------------------------------------------------------------
// 2) flatten_index_ignore_last_dim<RANK>:
//    Flatten all RANK dims in row-major order except ignoring dimension RANK-1.
template <uint32_t RANK>
inline uint32_t flatten_index_ignore_last_dim(const uint32_t (&multi_idx)[RANK], const uint32_t (&shape)[RANK]) {
    uint32_t offset = 0;
    for (uint32_t d = 0; d < RANK - 1; d++) {
        offset = offset * shape[d] + multi_idx[d];
    }
    return offset;
}

template <uint32_t RANK, uint32_t TILE_HEIGHT, uint32_t TILE_WIDTH>
inline uint32_t get_unpadded_linear_row_index_for_tile(
    uint32_t tile,
    const uint32_t (&input_tiled_shape)[RANK],  // [ ..., output_H_tiled, W_t ]
    const uint32_t (&input_shape)[RANK],        // [ ..., output_H,   W   ]
    uint32_t (&src_multi_idx)[RANK]) {
    // 1) Unflatten 'tile' => src_multi_idx in the tiled shape
    unflatten_index<RANK>(tile, input_tiled_shape, src_multi_idx);

    // 2) Multiply the output_H-dim by TILE_HEIGHT
    src_multi_idx[RANK - 2] *= TILE_HEIGHT;

    // 3) Flatten ignoring last dim => linear row offset
    return flatten_index_ignore_last_dim<RANK>(src_multi_idx, input_shape);
}

void kernel_main() {
    // ------------------------------------------------------------------------
    // 0) Read compile-time constants
    // ------------------------------------------------------------------------
    constexpr bool dst_is_dram = (get_compile_time_arg_val(0) == 1);
    constexpr uint32_t element_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(2);
    constexpr uint32_t output_H = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(6);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(7);
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(8);
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(9);
    constexpr bool needs_padding = (get_compile_time_arg_val(10) == 1);
    constexpr uint32_t RANK = get_compile_time_arg_val(11);
    constexpr uint32_t permuted_input_h_index = get_compile_time_arg_val(12);

    // ------------------------------------------------------------------------
    // 1) Read runtime arguments
    // ------------------------------------------------------------------------
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile = get_arg_val<uint32_t>(1);
    uint32_t end_tile = get_arg_val<uint32_t>(2);
    uint32_t start_padding_tile_idx = get_arg_val<uint32_t>(3);
    uint32_t end_padding_tile_idx = get_arg_val<uint32_t>(4);

    // input_shape, perm, output_shape
    uint32_t array_start_offset = 5;  // input shape starts at arg #5
    uint32_t input_shape[RANK], perm[RANK], output_shape[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + array_start_offset);
        perm[i] = get_arg_val<uint32_t>(i + array_start_offset + RANK);
    }
    for (uint32_t i = 0; i < RANK; i++) {
        output_shape[i] = input_shape[perm[i]];
    }

    // ------------------------------------------------------------------------
    // 2) Derived compile-time constants
    // ------------------------------------------------------------------------
    constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
    constexpr uint8_t NUM_FACES_H = TILE_HEIGHT / FACE_HEIGHT;
    constexpr uint8_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;

    // Padded dims
    constexpr uint32_t output_H_padded = tt::data_movement::common::round_up<output_H, TILE_HEIGHT>();
    constexpr uint32_t H_p = tt::data_movement::common::round_up<H, TILE_HEIGHT>();
    constexpr uint32_t W_p = tt::data_movement::common::round_up<W, TILE_WIDTH>();

    // Tiled dims
    constexpr uint32_t W_t = W_p / TILE_WIDTH;
    constexpr uint32_t H_t = H_p / TILE_HEIGHT;
    constexpr uint32_t output_H_tiled = output_H_padded / TILE_HEIGHT;

    // For sub-tile writes
    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_WIDTH * element_size;

    // Address generator
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const auto input_data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram, TILE_HW> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = input_data_format};

    // ------------------------------------------------------------------------
    // 3) Height dimension remainder logic
    // ------------------------------------------------------------------------
    constexpr uint32_t H_last_tile = H - (H_t - 1) * TILE_HEIGHT;
    uint8_t remainder_faces_h = tt::data_movement::common::div_up<H_last_tile, FACE_HEIGHT>();

    uint32_t remainder = H_last_tile % FACE_HEIGHT;
    uint8_t sub_tile_lines_real =
        (remainder == 0) ? static_cast<uint8_t>(FACE_HEIGHT) : static_cast<uint8_t>(remainder);

    // Some precomputed constants
    constexpr uint32_t face_height_width = FACE_HEIGHT * FACE_WIDTH;
    constexpr uint32_t face_h_stride = NUM_FACES_W * face_height_width;
    constexpr uint32_t face_height_width_bytes = face_height_width * element_size;
    constexpr uint32_t face_h_stride_bytes = face_h_stride * element_size;

    // ------------------------------------------------------------------------
    // 4) Build padded and tiled shapes
    // ------------------------------------------------------------------------
    uint32_t input_padded_shape[RANK];
    uint32_t input_tiled_shape[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        if (i < RANK - 2) {
            input_padded_shape[i] = input_shape[i];
            input_tiled_shape[i] = input_shape[i];
        } else if (i == RANK - 2) {
            input_padded_shape[i] = H_p;
            input_tiled_shape[i] = H_t;
        } else {
            // i == RANK - 1
            input_padded_shape[i] = W_p;
            input_tiled_shape[i] = W_t;
        }
    }

    uint32_t output_padded_shape[RANK];
    uint32_t output_tiled_shape[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        if (i < RANK - 2) {
            output_padded_shape[i] = output_shape[i];
            output_tiled_shape[i] = output_shape[i];
        } else if (i == RANK - 2) {
            output_padded_shape[i] = output_H_padded;
            output_tiled_shape[i] = output_H_tiled;
        } else {
            // i == RANK - 1
            output_padded_shape[i] = W_p;
            output_tiled_shape[i] = W_t;
        }
    }

    // ------------------------------------------------------------------------
    // 5) Build row strides for the destination padded shape
    // ------------------------------------------------------------------------
    uint32_t dest_padded_strides[RANK];
    dest_padded_strides[RANK - 1] = 1;
    dest_padded_strides[RANK - 2] = 1;  // dimension output_H in output
    for (int i = RANK - 3; i >= 0; i--) {
        dest_padded_strides[i] = dest_padded_strides[i + 1] * output_padded_shape[i + 1];
    }

    // ------------------------------------------------------------------------
    // 6) Main loop over all tiles [start_tile..end_tile)
    // ------------------------------------------------------------------------
    uint32_t src_multi_idx[RANK];
    uint32_t dest_multi_idx[RANK];

    for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
        // 6a) Unflatten 'tile' => src_multi_idx in input_tiled_shape
        unflatten_index<RANK>(tile, input_tiled_shape, src_multi_idx);

        uint32_t w_t_local = src_multi_idx[RANK - 1];  // tile index in W
        uint32_t h_t_local = src_multi_idx[RANK - 2];  // tile index in H

        // Determine how many faces in height have valid data
        uint8_t num_faces_h = (h_t_local == (H_t - 1)) ? remainder_faces_h : NUM_FACES_H;

        // Convert that tile's row dimension from tile index => row offset
        src_multi_idx[RANK - 2] *= TILE_HEIGHT;

        // Flatten => tile_start (the linear row offset in the unpadded shape)
        uint32_t tile_start = flatten_index_ignore_last_dim<RANK>(src_multi_idx, input_shape);

        // 6b) Build dest_multi_idx by permutation
        //     (the row dimension index for output tensor is also from src_multi_idx)
        for (uint32_t i = 0; i < RANK; ++i) {
            dest_multi_idx[i] = src_multi_idx[perm[i]];
        }

        // 6c) Compute base_output_row_offset ignoring the dimension permuted_input_h_index
        uint32_t base_output_row_offset = 0;
        for (uint32_t i = 0; i < RANK - 1; i++) {
            if (i == permuted_input_h_index) {
                continue;
            }
            base_output_row_offset += dest_multi_idx[i] * dest_padded_strides[i];
        }

        // This kernel is specialized so that permuted_input_h_index != RANK-2
        // => we can use RANK-2 for the tile dimension directly.
        uint32_t base_output_row_tile_start = dest_multi_idx[RANK - 2] % TILE_HEIGHT;

        // Face index in tile's height dimension
        uint32_t base_output_row_face_start = base_output_row_tile_start / FACE_HEIGHT;
        // Row within that face
        uint32_t output_sub_tile_line = base_output_row_tile_start % FACE_HEIGHT;

        // Precompute once for the tile:
        uint32_t output_face_line_offset =
            base_output_row_face_start * face_h_stride + output_sub_tile_line * FACE_WIDTH;
        // Also factor out the multiply by element_size once:
        uint32_t base_output_face_line_offset_bytes = output_face_line_offset * element_size;

        // 6d) Wait for data block
        cb_wait_front(cb_id_out0, 1);
        uint32_t base_l1_read_addr = get_read_ptr(cb_id_out0);

        // 6e) Loop over faces in the height dimension
        for (uint8_t face_h = 0; face_h < num_faces_h; ++face_h) {
            bool is_last_sub_tile_line = ((h_t_local == (H_t - 1)) && (face_h == num_faces_h - 1));
            uint8_t sub_tile_lines = is_last_sub_tile_line ? sub_tile_lines_real : FACE_HEIGHT;

            // row offset for the start of this face
            uint32_t base_row = tile_start + (face_h * FACE_HEIGHT);
            // also the offset in bytes for reading from L1
            uint32_t face_h_offset_bytes = face_h * face_h_stride_bytes;

            // 6f) For each line within that face
            for (uint8_t sub_tile_line = 0; sub_tile_line < sub_tile_lines; ++sub_tile_line) {
                // Compute the logical row
                uint32_t row = base_row + sub_tile_line;

                // Update src_multi_idx / dest_multi_idx for the row dimension only
                src_multi_idx[RANK - 2] = row % input_shape[RANK - 2];
                dest_multi_idx[permuted_input_h_index] = src_multi_idx[RANK - 2];

                // Flatten that row dimension into the output offset
                uint32_t output_row_offset = base_output_row_offset + dest_multi_idx[permuted_input_h_index] *
                                                                          dest_padded_strides[permuted_input_h_index];

                uint32_t output_tile_idx = (output_row_offset / TILE_HEIGHT) * W_t + w_t_local;

                // 6g) Loop over faces in the width dimension
                for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                    // face_w offset in bytes
                    uint32_t face_w_offset_bytes = face_w * face_height_width_bytes;

                    // Where data goes in the *output* tile
                    uint32_t output_tile_offset_bytes = base_output_face_line_offset_bytes + face_w_offset_bytes;

                    // Build final output address
                    uint64_t write_noc_base_addr = get_noc_addr(output_tile_idx, s, output_tile_offset_bytes);

                    // Build final input read address
                    uint32_t final_addr = base_l1_read_addr + face_h_offset_bytes + face_w_offset_bytes +
                                          (sub_tile_line * SUBTILE_LINE_BYTES);

                    // 6h) Asynchronous write
                    noc_async_write(final_addr, write_noc_base_addr, SUBTILE_LINE_BYTES);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }

    // ------------------------------------------------------------------------
    // 7) Handle padding if needed
    // ------------------------------------------------------------------------
    if constexpr (needs_padding) {
        cb_wait_front(tt::CBIndex::c_1, 1);
        uint32_t l1_read_ptr = get_read_ptr(tt::CBIndex::c_1);

        // We'll reuse 'dest_multi_idx' for tile indexing
        constexpr uint32_t x_t = output_H_tiled - 1;
        constexpr uint8_t X_in_tile = output_H % TILE_HEIGHT;
        constexpr uint8_t face_c_start = (X_in_tile / FACE_HEIGHT);

        dest_multi_idx[RANK - 2] = x_t;  // fix the tile dimension in output

        for (uint32_t tile_idx = start_padding_tile_idx; tile_idx < end_padding_tile_idx; ++tile_idx) {
            // Unflatten 'tile_idx' => dest_multi_idx in the output tiled shape
            size_t remaining = tile_idx;
            for (uint32_t i = 0; i < RANK; ++i) {
                size_t dim = RANK - 1 - i;
                if (dim == (RANK - 2)) {
                    continue;  // already set x_t
                }
                dest_multi_idx[dim] = (remaining % output_tiled_shape[dim]);
                remaining /= output_tiled_shape[dim];
            }

            // Flatten => linear_idx
            uint32_t linear_idx = 0;
            for (uint32_t i = 0; i < RANK; ++i) {
                linear_idx = (linear_idx * output_tiled_shape[i]) + dest_multi_idx[i];
            }

            // Write out padding lines
            for (uint8_t face_c = face_c_start; face_c < NUM_FACES_H; ++face_c) {
                uint32_t face_c_offset = face_c * NUM_FACES_W * face_height_width;
                uint8_t sub_tile_line_start = (face_c == face_c_start) ? (X_in_tile % FACE_HEIGHT) : 0;

                for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                    uint32_t face_offset = face_c_offset + (face_w * face_height_width);

                    for (uint8_t sub_tile_line = sub_tile_line_start; sub_tile_line < FACE_HEIGHT; ++sub_tile_line) {
                        uint32_t offset = (face_offset + (sub_tile_line * FACE_WIDTH)) * element_size;

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
