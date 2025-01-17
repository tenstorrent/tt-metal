// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

template <uint32_t N>
void dprint_array(const uint32_t* arr, const char* name) {
    DPRINT << name << ": ";
    for (uint32_t i = 0; i < N; i++) {
        DPRINT << arr[i] << " ";
    }
    DPRINT << ENDL();
}

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

void kernel_main() {
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t element_size = get_compile_time_arg_val(3);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(4);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(5);
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(6);
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(7);
    constexpr uint32_t x_dim = get_compile_time_arg_val(8);
    constexpr uint32_t X = get_compile_time_arg_val(9);
    constexpr uint32_t W = get_compile_time_arg_val(10);
    constexpr uint32_t output_H = get_compile_time_arg_val(11);
    constexpr uint32_t X_p = get_compile_time_arg_val(12);
    constexpr uint32_t W_p = get_compile_time_arg_val(13);
    constexpr uint32_t non_x_rows = get_compile_time_arg_val(14);
    // constexpr uint32_t output_H_t = get_compile_time_arg_val(15);
    constexpr uint32_t W_t = get_compile_time_arg_val(16);
    constexpr uint32_t final_tile_real_x = get_compile_time_arg_val(17);
    constexpr uint32_t final_tile_real_faces_x = get_compile_time_arg_val(18);
    constexpr uint32_t xw_blocks = get_compile_time_arg_val(19);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(20);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(21);
    constexpr bool needs_y_padding = (bool)get_compile_time_arg_val(22);
    constexpr uint32_t permuted_w_dim = get_compile_time_arg_val(23);

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
    // W dimension is always the last dimension
    constexpr uint32_t w_dim = N - 1;
    constexpr uint32_t H_p = TILE_HEIGHT * ((output_H + TILE_HEIGHT - 1) / TILE_HEIGHT);
    constexpr uint32_t output_H_t = H_p / TILE_HEIGHT;
    constexpr uint32_t X_t = X_p / TILE_HEIGHT;
    constexpr uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;

    constexpr uint32_t cb_out = tt::CBIndex::c_2;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block = get_arg_val<uint32_t>(1);
    uint32_t end_block = get_arg_val<uint32_t>(2);
    uint32_t start_padding_tile_idx = get_arg_val<uint32_t>(3);
    uint32_t end_padding_tile_idx = get_arg_val<uint32_t>(4);

    uint32_t array_start_offset = 5;
    uint32_t input_shape[N], dims[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + array_start_offset);
        dims[i] = get_arg_val<uint32_t>(i + N + array_start_offset);
    }

    uint32_t output_shape[N];
    for (uint32_t i = 0; i < N; i++) {
        output_shape[i] = input_shape[dims[i]];
    }

    // ------------------------------------------------------------------------
    // Build padded and tiled shapes
    // ------------------------------------------------------------------------
    uint32_t output_tiled_shape[N];
    for (uint32_t i = 0; i < N; i++) {
        if (i < N - 2) {
            output_tiled_shape[i] = output_shape[i];
        } else if (i == N - 2) {
            output_tiled_shape[i] = output_H_t;
        } else {
            // i == N - 1
            output_tiled_shape[i] = X_t;
        }
    }

    // ------------------------------------------------------------------------
    // Build row strides for the destination padded shape
    // ------------------------------------------------------------------------
    uint32_t dest_tiled_strides[N];

    dest_tiled_strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) {
        dest_tiled_strides[i] = dest_tiled_strides[i + 1] * output_tiled_shape[i + 1];
    }
    uint32_t W_stride_tile = dest_tiled_strides[permuted_w_dim];

    DPRINT << "start_block: " << start_block << ENDL();
    DPRINT << "end_block: " << end_block << ENDL();
    DPRINT << "N: " << N << ENDL();
    DPRINT << "input_cb_page_size: " << input_cb_page_size << ENDL();
    DPRINT << "element_size: " << element_size << ENDL();
    DPRINT << "TILE_HEIGHT: " << TILE_HEIGHT << ENDL();
    DPRINT << "TILE_WIDTH: " << TILE_WIDTH << ENDL();
    DPRINT << "FACE_HEIGHT: " << FACE_HEIGHT << ENDL();
    DPRINT << "FACE_WIDTH: " << FACE_WIDTH << ENDL();
    DPRINT << "dst_addr: " << dst_addr << ENDL();
    DPRINT << "start_block: " << start_block << ENDL();
    DPRINT << "end_block: " << end_block << ENDL();
    DPRINT << "x_dim: " << x_dim << ENDL();
    DPRINT << "X: " << X << ENDL();
    DPRINT << "W: " << W << ENDL();
    DPRINT << "X_p: " << X_p << ENDL();
    DPRINT << "W_p: " << W_p << ENDL();
    DPRINT << "xw_blocks: " << xw_blocks << ENDL();
    DPRINT << "x_block_size: " << x_block_size << ENDL();
    DPRINT << "w_block_size: " << w_block_size << ENDL();
    DPRINT << "w_blocks: " << w_blocks << ENDL();
    DPRINT << "x_blocks: " << x_blocks << ENDL();
    DPRINT << "final_tile_real_x: " << final_tile_real_x << ENDL();
    DPRINT << "final_tile_real_faces_x: " << final_tile_real_faces_x << ENDL();
    DPRINT << "W_stride_tile: " << W_stride_tile << ENDL();

    dprint_array<N>(input_shape, "input_shape");
    dprint_array<N>(dims, "dims");
    dprint_array<N>(output_shape, "output_shape");
    dprint_array<N>(output_tiled_shape, "output_tiled_shape");
    dprint_array<N>(dest_tiled_strides, "dest_tiled_strides");

    /**
     * We have a multidimensional tensor:
     * - num_blocks_total = (rows * x_blocks * w_blocks) where rows = num_rows / X
     *   Here, 'rows' represent the combination of all rows before and after X dimension.
     *   So: rows * X * W_dimension = total number of elements (conceptually).
     *
     * For each 'block':
     *   - Compute which w_block and x_block this corresponds to.
     *   - Then compute which row set (xw_block) we are in.
     */

    // x_dim is the dimension along which we are reading the tensor, as it's the new W dimension in the output tensor
    // Find where the original W dimension ended up in the permuted output

    const DataFormat data_format = get_dataformat(tt::CBIndex::c_0);
    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t idxs[N];
    idxs[N - 1] = 0;
    uint32_t dest_multi_idx[N];
    uint32_t output_h = 0;
    uint8_t output_sub_tile_line = 0;
    uint8_t output_face_h = 0;
    uint16_t base_face_line_offset_bytes = 0;
    for (uint32_t block = start_block; block < end_block; block++) {
        // Decompose block into w_block, x_block, and xw_block indices
        uint32_t rem = block;
        const uint32_t w_block = rem % w_blocks;  // Which W block are we in?
        rem /= w_blocks;

        const uint32_t x_block = rem % x_blocks;  // Which X block?
        rem /= x_blocks;

        uint32_t xw_block = rem % (non_x_rows);  // Which row set (beyond X dimension)?
        uint32_t remainder = xw_block;

        // Compute X block boundaries
        uint32_t x_start = x_block * x_block_size;

        // Compute W block boundaries
        uint32_t w_start = w_block * w_block_size;
        uint32_t w_end = min(w_start + w_block_size, input_shape[N - 1]);

        // Map linear index i to multidimensional indices idxs[]
        // We skip x_dim when doing this mapping and set it separately later
        for (int32_t d = N - 2; d >= 0; --d) {  // Exclude W as w_block already equals w_t
            if (d == (int32_t)x_dim) {
                idxs[d] = 0;  // Initialize x_dim to zero (will be set in inner loop)
                continue;     // Skip x_dim during mapping
            }
            idxs[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }
        idxs[N - 1] = w_start;  // Initialize W dimension index to zero if not already set
        for (uint32_t d = 0; d < N; ++d) {
            dest_multi_idx[d] = idxs[dims[d]];
        }
        dest_multi_idx[permuted_w_dim] = w_start;
        if constexpr (permuted_w_dim != N - 2) {
            output_h = dest_multi_idx[N - 2];
            output_sub_tile_line = output_h % FACE_HEIGHT;
            output_face_h = (output_h % TILE_HEIGHT) / FACE_HEIGHT;
            base_face_line_offset_bytes =
                output_face_h * FACE_H_STRIDE_BYTES + output_sub_tile_line * SUBTILE_LINE_BYTES;
        }

        dest_multi_idx[N - 2] /=
            TILE_HEIGHT;  // if permuted_w_dim is w_start, then this is still the correct tile index

        dest_multi_idx[N - 1] = x_block;
        // Compute final linear index for the current W
        uint32_t base_tile_offset = 0;
        for (uint32_t i = 0; i < N; ++i) {
            if constexpr (permuted_w_dim != N - 2) {
                if (i == permuted_w_dim) {
                    continue;
                }
            }
            base_tile_offset += dest_multi_idx[i] * dest_tiled_strides[i];
        }

        cb_wait_front(cb_out, 1);
        uint32_t transposed_buffer_read_addr = get_read_ptr(cb_out);
        uint32_t real_faces_x = x_block != X_t - 1 ? NUM_FACES_W : final_tile_real_faces_x;
        // Iterate over the W dimension elements
        for (uint32_t w = w_start; w < w_end; ++w) {
            // Compute final linear tile index for the current W
            uint32_t tile = 0;
            if constexpr (permuted_w_dim != N - 2) {
                tile = base_tile_offset + w * W_stride_tile;
            } else {
                output_sub_tile_line = w % FACE_HEIGHT;
                output_face_h = (w % TILE_HEIGHT) / FACE_HEIGHT;
                base_face_line_offset_bytes =
                    output_face_h * FACE_H_STRIDE_BYTES + output_sub_tile_line * SUBTILE_LINE_BYTES;
                tile = base_tile_offset;
            }

            // Compute the NoC address for the output
            uint16_t page_offset = (w - w_start) * TILE_LINE_BYTES;

            uint64_t dest_noc_addr = get_noc_addr(tile, s, base_face_line_offset_bytes);

            for (uint8_t i = 0; i < real_faces_x; i++) {
                uint16_t w_offset = i * FACE_HW_BYTES;
                uint16_t cb_w_offset = i * SUBTILE_LINE_BYTES;
                noc_async_write(
                    transposed_buffer_read_addr + cb_w_offset + page_offset,
                    dest_noc_addr + w_offset,
                    SUBTILE_LINE_BYTES);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }

    // add padding
    if constexpr (needs_y_padding) {
        cb_wait_front(tt::CBIndex::c_3, 1);
        uint32_t l1_read_ptr = get_read_ptr(tt::CBIndex::c_3);

        // We'll reuse 'dest_multi_idx' for tile indexing
        constexpr uint32_t y_t = output_H_t - 1;
        uint8_t Y_in_tile = output_shape[N - 2] % TILE_HEIGHT;
        uint8_t face_y_start = (Y_in_tile / FACE_HEIGHT);

        dest_multi_idx[N - 2] = y_t;  // fix the tile dimension in output

        for (uint32_t tile_idx = start_padding_tile_idx; tile_idx < end_padding_tile_idx; ++tile_idx) {
            // Unflatten 'tile_idx' => dest_multi_idx in the output tiled shape
            size_t remaining = tile_idx;
            for (uint32_t i = 0; i < N; ++i) {
                size_t dim = N - 1 - i;
                if (dim == (N - 2)) {
                    continue;  // already set y_t
                }
                dest_multi_idx[dim] = (remaining % output_tiled_shape[dim]);
                remaining /= output_tiled_shape[dim];
            }

            // Flatten => linear_idx
            uint32_t linear_idx = 0;
            for (uint32_t i = 0; i < N; ++i) {
                linear_idx = (linear_idx * output_tiled_shape[i]) + dest_multi_idx[i];
            }

            // Write out padding lines
            for (uint8_t face_y = face_y_start; face_y < NUM_FACES_H; ++face_y) {
                uint32_t face_y_offset = face_y * NUM_FACES_W * FACE_HW;
                uint8_t sub_tile_line_start = (face_y == face_y_start) ? (Y_in_tile % FACE_HEIGHT) : 0;

                for (uint8_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                    uint32_t face_offset = face_y_offset + (face_w * FACE_HW);

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
        cb_pop_front(tt::CBIndex::c_3, 1);
    }
}
