// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // ------------------------------------------------------------------------
    // 1) Compile-time arguments
    // ------------------------------------------------------------------------
    constexpr bool src0_is_dram = static_cast<bool>(get_compile_time_arg_val(0));
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
    constexpr bool needs_x_padding = static_cast<bool>(get_compile_time_arg_val(24));
    constexpr bool needs_y_padding = static_cast<bool>(get_compile_time_arg_val(25));
    constexpr uint32_t non_x_rows = get_compile_time_arg_val(26);
    constexpr uint32_t read_alignment = get_compile_time_arg_val(27);

    // ------------------------------------------------------------------------
    // 2) Derived Constants (kept as constexpr)
    // ------------------------------------------------------------------------
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

    constexpr uint32_t misalignment = read_alignment - SUBTILE_LINE_BYTES;

    // For x-padding logic:
    constexpr uint32_t final_face_real_w = (W % FACE_WIDTH);
    constexpr uint32_t ratio = sizeof(uint32_t) / element_size;
    // If the last tile in W dimension is partially filled, we might need fewer writes
    constexpr uint32_t final_x_pad_write =
        (final_face_real_w == 0) ? num_writes : (final_face_real_w + ratio - 1) / ratio;

    // ------------------------------------------------------------------------
    // 3) Runtime Args
    // ------------------------------------------------------------------------
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block = get_arg_val<uint32_t>(1);
    uint32_t end_block = get_arg_val<uint32_t>(2);

    uint32_t input_shape[N], dims[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(i + 3);
        dims[i] = get_arg_val<uint32_t>(i + N + 3);
    }

    // ------------------------------------------------------------------------
    // 4) Build padded/tiled shapes & strides
    // ------------------------------------------------------------------------
    uint32_t input_tiled_shape[N];
    for (uint32_t i = 0; i < N; i++) {
        if (i < N - 2) {
            input_tiled_shape[i] = input_shape[i];
        } else if (i == N - 2) {
            input_tiled_shape[i] = H_t;
        } else {
            input_tiled_shape[i] = W_t;  // i == N - 1
        }
    }

    uint32_t src_tiled_strides[N];
    src_tiled_strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) {
        src_tiled_strides[i] = src_tiled_strides[i + 1] * input_tiled_shape[i + 1];
    }
    constexpr auto data_format = get_dataformat(tt::CBIndex::c_0);
    const InterleavedAddrGenFast<src0_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // Stride for stepping along x_dim
    const uint32_t X_stride_tile = src_tiled_strides[x_dim];

    // We'll map our multi-dim index into 'idxs'
    uint32_t idxs[N];
    idxs[N - 1] = 0;

    // Decide how many faces to read in final W block
    // (If final_tile_real_faces_w < NUM_FACES_W, we have partial data in the last block.)
    uint32_t real_faces_w = 0;
    if constexpr (NUM_FACES_W == final_tile_real_faces_w) {
        // Full tile all the time
        real_faces_w = NUM_FACES_W;
    }

    // ------------------------------------------------------------------------
    // 5) Main loop over [start_block..end_block)
    // ------------------------------------------------------------------------
    for (uint32_t block = start_block; block < end_block; ++block) {
        // Decompose block => w_block, x_block, h, xw_block
        uint32_t rem = block;
        const uint32_t w_block = rem % w_blocks;
        rem /= w_blocks;

        const uint32_t x_block = rem % x_blocks;
        rem /= x_blocks;

        uint32_t h = rem % H;
        // Next is sub-tile offsets
        uint32_t sub_tile_line = h % FACE_HEIGHT;
        uint32_t face_h = (h % TILE_HEIGHT) / FACE_HEIGHT;
        uint32_t base_face_line_offset_bytes = face_h * FACE_H_STRIDE_BYTES + sub_tile_line * SUBTILE_LINE_BYTES;

        uint32_t xw_block = rem % non_x_rows;
        uint32_t remainder = xw_block;

        // X range for this block
        uint32_t x_start = x_block * x_block_size;
        uint32_t x_end = (x_start + x_block_size < X) ? (x_start + x_block_size) : X;

        // Fill idxs[] except for x_dim
        for (int32_t d = N - 2; d >= 0; --d) {
            if (d == (int32_t)x_dim) {
                idxs[d] = 0;
                continue;
            }
            idxs[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }
        idxs[N - 1] = w_block;

        // Compute tile offset ignoring x_dim
        uint64_t base_tile_offset = 0;
        for (uint32_t d = 0; d < N; d++) {
            if (d == x_dim) {
                continue;
            }
            uint32_t tile_index = (d == (N - 2)) ? (idxs[d] / TILE_HEIGHT) : idxs[d];
            base_tile_offset += static_cast<uint64_t>(tile_index) * src_tiled_strides[d];
        }

        // If the final W-block can be partial, figure out how many faces
        if constexpr (NUM_FACES_W != final_tile_real_faces_w) {
            real_faces_w = (w_block == (W_t - 1)) ? final_tile_real_faces_w : NUM_FACES_W;
        }

        // Reserve a slot in the circular buffer, get L1 pointer
        cb_reserve_back(tt::CBIndex::c_0, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
        if constexpr (misalignment > 0) {
            if ((h & 1) == 1) {
                src_buffer_l1_addr += misalignment;
            }
        }

        // --------------------------------------------------------------------
        // 5.1) Async read for [x_start..x_end)
        // --------------------------------------------------------------------
        {
            uint64_t tile = base_tile_offset + (static_cast<uint64_t>(x_start) * X_stride_tile);
            uint32_t page_offset = 0;

            for (uint32_t x = x_start; x < x_end; ++x) {
                uint32_t l1_col_base = src_buffer_l1_addr + page_offset;
                uint64_t src_noc_addr = get_noc_addr(tile, s, base_face_line_offset_bytes);

                // Read each face in [0..real_faces_w)
                uint16_t w_offset = 0;
                uint16_t cb_w_offset = 0;
                for (uint8_t i = 0; i < real_faces_w; i++) {
                    noc_async_read(src_noc_addr + w_offset, l1_col_base + cb_w_offset, SUBTILE_LINE_BYTES);

                    w_offset += FACE_HW_BYTES;
                    cb_w_offset += SUBTILE_LINE_BYTES;
                }
                tile += X_stride_tile;
                page_offset += TILE_LINE_BYTES;
            }
        }

        // --------------------------------------------------------------------
        // 5.2) X Padding if needed
        // --------------------------------------------------------------------
        if constexpr (needs_x_padding) {
            // If this is the last X-block, fill leftover space up to X_p
            if (x_block == x_blocks - 1) {
                // We'll keep track of page_offset for the padding region
                uint32_t padding_page_offset = (x_end - x_start) * TILE_LINE_BYTES;

                for (uint32_t px = x_end; px < X_p; ++px) {
                    uint32_t l1_col_base = src_buffer_l1_addr + padding_page_offset;

                    for (uint8_t i = 0; i < real_faces_w; i++) {
                        // Possibly differentiate final_x_pad_write vs num_writes
                        // if the final face is partial. E.g.:
                        if constexpr (num_writes != final_x_pad_write) {
                            // If we truly need that logic:
                            bool last_face_in_final_wblock = (w_block == (w_blocks - 1)) && (i == real_faces_w - 1);
                            uint32_t writes = last_face_in_final_wblock ? final_x_pad_write : num_writes;
                            tt::data_movement::common::fill_with_val(
                                l1_col_base + i * SUBTILE_LINE_BYTES, writes, padding_val_packed);
                        } else {
                            // simpler route if final_x_pad_write == num_writes
                            tt::data_movement::common::fill_with_val(
                                l1_col_base + i * SUBTILE_LINE_BYTES, num_writes, padding_val_packed);
                        }
                    }
                    padding_page_offset += TILE_LINE_BYTES;
                }
            }
        }

        // Wait for reads to complete, push the tile
        noc_async_read_barrier();
        cb_push_back(tt::CBIndex::c_0, 1);
    }

    // ------------------------------------------------------------------------
    // 6) Y Padding
    // ------------------------------------------------------------------------
    if constexpr (needs_y_padding) {
        // We store one chunk of padding in c_3
        cb_reserve_back(tt::CBIndex::c_3, 1);
        uint32_t l1_write_addr = get_write_ptr(tt::CBIndex::c_3);
        tt::data_movement::common::fill_with_val(l1_write_addr, num_writes, padding_val_packed);
        cb_push_back(tt::CBIndex::c_3, 1);
    }
}
