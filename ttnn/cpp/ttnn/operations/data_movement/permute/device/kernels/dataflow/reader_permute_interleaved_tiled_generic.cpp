// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

// template <uint32_t N>
// void dprint_array(const uint32_t* arr, const char* name) {
//     DPRINT << name << ": ";
//     for (uint32_t i = 0; i < N; i++) {
//         DPRINT << arr[i] << " ";
//     }
//     DPRINT << ENDL();
// }

void kernel_main() {
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(0);
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
    constexpr uint32_t H = get_compile_time_arg_val(11);
    constexpr uint32_t X_p = get_compile_time_arg_val(12);
    constexpr uint32_t W_p = get_compile_time_arg_val(13);
    constexpr uint32_t H_p = get_compile_time_arg_val(14);
    constexpr uint32_t H_t = get_compile_time_arg_val(15);
    constexpr uint32_t W_t = get_compile_time_arg_val(16);
    constexpr uint32_t final_tile_real_w = get_compile_time_arg_val(17);
    constexpr uint32_t final_tile_real_faces_w = get_compile_time_arg_val(18);
    constexpr uint32_t xw_blocks = get_compile_time_arg_val(19);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(20);
    constexpr uint32_t w_blocks = get_compile_time_arg_val(21);
    constexpr uint32_t num_writes = get_compile_time_arg_val(22);
    constexpr uint32_t padding_val_packed = get_compile_time_arg_val(23);
    constexpr bool needs_x_padding = (bool)get_compile_time_arg_val(24);
    constexpr bool needs_y_padding = (bool)get_compile_time_arg_val(25);
    constexpr uint32_t non_x_rows = get_compile_time_arg_val(26);

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
    constexpr uint32_t final_face_real_w = W % FACE_WIDTH;

    constexpr uint32_t ratio = sizeof(uint32_t) / element_size;
    constexpr uint32_t final_x_pad_write =
        final_face_real_w == 0 ? num_writes : (final_face_real_w + ratio - 1) / ratio;
    constexpr uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block = get_arg_val<uint32_t>(1);
    uint32_t end_block = get_arg_val<uint32_t>(2);

    // Input shape and dims
    uint32_t input_shape[N], dims[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + 3);
        dims[i] = get_arg_val<uint32_t>(i + N + 3);
    }

    // ------------------------------------------------------------------------
    // 4) Build padded and tiled shapes
    // ------------------------------------------------------------------------
    uint32_t input_tiled_shape[N];
    for (uint32_t i = 0; i < N; i++) {
        if (i < N - 2) {
            input_tiled_shape[i] = input_shape[i];
        } else if (i == N - 2) {
            input_tiled_shape[i] = H_t;
        } else {
            // i == N - 1
            input_tiled_shape[i] = W_t;
        }
    }

    // ------------------------------------------------------------------------
    // 5) Build row strides for the destination padded shape
    // ------------------------------------------------------------------------
    uint32_t src_tiled_strides[N];
    src_tiled_strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) {
        src_tiled_strides[i] = src_tiled_strides[i + 1] * input_tiled_shape[i + 1];
    }

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
    uint32_t X_stride_tile = src_tiled_strides[x_dim];

    const DataFormat data_format = get_dataformat(tt::CBIndex::c_0);
    const InterleavedAddrGenFast<src0_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t idxs[N];
    idxs[N - 1] = 0;

    uint32_t real_faces_w = 0;
    if constexpr (NUM_FACES_W == final_tile_real_faces_w) {
        real_faces_w = NUM_FACES_W;
    }
    for (uint32_t block = start_block; block < end_block; ++block) {
        // Decompose block into w_block, x_block, and xw_block indices
        uint32_t rem = block;
        const uint32_t w_block = rem % w_blocks;  // Which W block are we in?
        rem /= w_blocks;

        const uint32_t x_block = rem % x_blocks;  // Which X block?
        rem /= x_blocks;

        uint32_t h = rem % H;
        uint32_t sub_tile_line = h % FACE_HEIGHT;
        uint32_t face_h = (h % TILE_HEIGHT) / FACE_HEIGHT;
        uint32_t base_face_line_offset_bytes = face_h * FACE_H_STRIDE_BYTES + sub_tile_line * SUBTILE_LINE_BYTES;

        uint32_t xw_block = rem % (non_x_rows);  // Which row set (beyond X dimension)?
        uint32_t remainder = xw_block;

        // Compute X block boundaries
        uint32_t x_start = x_block * x_block_size;
        uint32_t x_end = min(x_start + x_block_size, X);

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
        idxs[N - 1] = w_block;

        // Precompute the base address offset (excluding x_dim)
        uint64_t base_tile_offset = 0;
        for (uint32_t d = 0; d < N; ++d) {
            if (d == x_dim) {
                continue;
            }
            uint32_t index = d != N - 2 ? idxs[d] : idxs[d] / TILE_HEIGHT;
            base_tile_offset += index * src_tiled_strides[d];
        }
        if constexpr (NUM_FACES_W != final_tile_real_faces_w) {
            real_faces_w = w_block != W_t - 1 ? NUM_FACES_W : final_tile_real_faces_w;
        }

        // Reserve space in the circular buffer for the X-block length
        cb_reserve_back(tt::CBIndex::c_0, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
        // We read in 'x_block_len' chunks along the X dimension
        // Read along the X dimension
        for (uint32_t x = x_start; x < x_end; ++x) {
            uint32_t tile = base_tile_offset + x * X_stride_tile;
            // Compute the address offset for this index
            // Build final output address
            uint16_t page_offset = (x - x_start) * TILE_LINE_BYTES;
            uint64_t src_noc_addr = get_noc_addr(tile, s, base_face_line_offset_bytes);
            for (uint8_t i = 0; i < real_faces_w; i++) {
                uint16_t w_offset = i * FACE_HW_BYTES;
                uint16_t cb_w_offset = i * SUBTILE_LINE_BYTES;
                noc_async_read(
                    src_noc_addr + w_offset, src_buffer_l1_addr + page_offset + cb_w_offset, SUBTILE_LINE_BYTES);
            }
        }
        if constexpr (needs_x_padding) {
            // final x_block needs padding
            if (x_block == x_blocks - 1) {
                uint32_t writes = 0;
                if constexpr (num_writes == final_x_pad_write) {
                    writes = num_writes;
                }
                for (uint32_t x = x_end; x < X_p; ++x) {
                    // Compute the address offset for this index
                    // Build final output address
                    uint32_t page_offset = (x - x_start) * TILE_LINE_BYTES;
                    for (uint8_t i = 0; i < real_faces_w; i++) {
                        uint32_t cb_w_offset = i * SUBTILE_LINE_BYTES;
                        if constexpr (num_writes != final_x_pad_write) {
                            writes =
                                (w_block == w_blocks - 1 && i == real_faces_w - 1) ? final_x_pad_write : num_writes;
                        }
                        tt::data_movement::common::fill_with_val(
                            src_buffer_l1_addr + page_offset + cb_w_offset, writes, padding_val_packed);
                    }
                }
            }
        }

        // Wait for all async reads to complete before proceeding
        noc_async_read_barrier();

        // Push the filled block into the circular buffer
        cb_push_back(tt::CBIndex::c_0, 1);
    }
    if constexpr (needs_y_padding) {
        // Add padding
        cb_reserve_back(tt::CBIndex::c_3, 1);
        uint32_t l1_write_addr = get_write_ptr(tt::CBIndex::c_3);
        // Fill with padding value
        tt::data_movement::common::fill_with_val(l1_write_addr, num_writes, padding_val_packed);
        cb_push_back(tt::CBIndex::c_3, 1);
    }
}
