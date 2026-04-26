// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t rank = get_named_compile_time_arg_val("rank");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");
    constexpr uint32_t TILE_H = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t TILE_W = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t FACE_H = get_named_compile_time_arg_val("face_height");
    constexpr uint32_t FACE_W = get_named_compile_time_arg_val("face_width");
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t NUM_FACES_H = TILE_H / FACE_H;
    constexpr uint32_t NUM_FACES_W = TILE_W / FACE_W;
    constexpr uint32_t NUM_FACES = NUM_FACES_H * NUM_FACES_W;
    constexpr uint32_t FACE_HW_BYTES = FACE_H * FACE_W * element_size;
    constexpr uint32_t SUBTILE_ROW_BYTES = FACE_W * element_size;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile = get_arg_val<uint32_t>(2);

    uint32_t tiled_shape[rank], tile_strides[rank], dims_to_flip[rank];
    for (uint32_t i = 0; i < rank; i++) {
        tiled_shape[i] = get_arg_val<uint32_t>(i + 3);
        tile_strides[i] = get_arg_val<uint32_t>(i + rank + 3);
        dims_to_flip[i] = get_arg_val<uint32_t>(i + rank + rank + 3);
    }

    const bool is_vertical_flip = static_cast<bool>(dims_to_flip[rank - 2]);
    const bool is_horizontal_flip = static_cast<bool>(dims_to_flip[rank - 1]);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const uint32_t tile_size = get_tile_size(cb_id);
    const auto s0 = TensorAccessor(src_args, src_addr, tile_size);

    // face layout: top-left=0, top-right=1, bottom-left=2, bottom-right=3
    // reordering faces handles the tile-level flip for h/w dims
    static const uint32_t face_order[4][NUM_FACES] = {
        {0, 1, 2, 3},
        {1, 0, 3, 2},  // horizontal
        {2, 3, 0, 1},  // vertical
        {3, 2, 1, 0},  // both
    };
    const uint32_t* face_read_order = face_order[(is_horizontal_flip ? 1 : 0) + (is_vertical_flip ? 2 : 0)];

    // initialize odometer from start_tile using div/mod once
    uint32_t dst_multi_dim[rank];
    uint32_t rem = start_tile;
    for (uint32_t i = 0; i < rank; i++) {
        dst_multi_dim[i] = rem / tile_strides[i];
        rem = rem % tile_strides[i];
    }

    for (uint32_t tile_id = start_tile; tile_id < end_tile; tile_id++) {
        uint32_t src_tile_id = 0;
        for (uint32_t i = 0; i < rank; i++) {
            uint32_t dim = dims_to_flip[i] ? (tiled_shape[i] - dst_multi_dim[i] - 1) : dst_multi_dim[i];
            src_tile_id += dim * tile_strides[i];
        }

        cb_reserve_back(cb_id, 1);
        uint32_t l1_addr = get_write_ptr(cb_id);
        noc_async_read_tile(src_tile_id, s0, l1_addr);
        noc_async_read_barrier();

        uint8_t* tile_ptr = reinterpret_cast<uint8_t*>(l1_addr);
        for (uint32_t fi = 0; fi < NUM_FACES; fi++) {
            uint8_t* face_ptr = tile_ptr + face_read_order[fi] * FACE_HW_BYTES;

            int32_t row_start = is_vertical_flip ? (int32_t)FACE_H - 1 : 0;
            int32_t row_end = is_vertical_flip ? -1 : (int32_t)FACE_H;
            int32_t row_step = is_vertical_flip ? -1 : 1;

            for (int32_t row = row_start; row != row_end; row += row_step) {
                uint8_t* row_ptr = face_ptr + row * SUBTILE_ROW_BYTES;
                if (is_horizontal_flip) {
                    for (uint32_t col = 0; col < FACE_W / 2; col++) {
                        uint32_t left = col * element_size;
                        uint32_t right = (FACE_W - 1 - col) * element_size;
                        for (uint32_t b = 0; b < element_size; b++) {
                            uint8_t tmp = row_ptr[left + b];
                            row_ptr[left + b] = row_ptr[right + b];
                            row_ptr[right + b] = tmp;
                        }
                    }
                }
            }
        }

        cb_push_back(cb_id, 1);

        for (int j = (int)rank - 1; j >= 0; j--) {
            if (++dst_multi_dim[j] < tiled_shape[j]) {
                break;
            }
            dst_multi_dim[j] = 0;
        }
    }
}
