// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

inline uint32_t calc_src_tile_index(
    uint32_t dst_tile_id, uint32_t rank, uint32_t* dims_to_flip, uint32_t* tiled_shape, uint32_t* tile_strides) {
    size_t remaining = dst_tile_id;
    uint32_t src_multi_dim[rank];  // TODO: do not use VLAs
    uint32_t dst_multi_dim[rank];  // TODO: do not use VLAs

    // 1. Convert output tile linear index to multi-dimensional index
    for (uint32_t i = 0; i < rank; ++i) {
        uint32_t dim = rank - 1 - i;
        dst_multi_dim[dim] = remaining % tiled_shape[dim];
        remaining /= tiled_shape[dim];
    }

    // 2. Based on 1) compute multi-dimensional index for the source tile
    for (uint32_t i = 0; i < rank; ++i) {
        if (dims_to_flip[i]) {
            src_multi_dim[i] = tiled_shape[i] - dst_multi_dim[i] - 1;
        } else {
            src_multi_dim[i] = dst_multi_dim[i];
        }
    }

    // 3. Convert source tile multi-dimensional index to linear index
    uint32_t src_tile_id = 0;
    for (uint32_t i = 0; i < rank; ++i) {
        src_tile_id += src_multi_dim[i] * tile_strides[i];
    }
    return src_tile_id;
}

void kernel_main() {
    // Compile time arguments
    constexpr bool src_is_dram = static_cast<bool>(get_named_compile_time_arg_val("src_is_dram"));
    constexpr uint32_t RANK = get_named_compile_time_arg_val("rank");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");
    constexpr uint32_t TILE_HEIGHT = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t TILE_WIDTH = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t FACE_HEIGHT = get_named_compile_time_arg_val("face_height");
    constexpr uint32_t FACE_WIDTH = get_named_compile_time_arg_val("face_width");
    constexpr auto src_args = TensorAccessorArgs<0>();

    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile = get_arg_val<uint32_t>(2);

    uint32_t tiled_shape[RANK], tile_strides[RANK], dims_to_flip[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        tiled_shape[i] = get_arg_val<uint32_t>(i + 3);
        tile_strides[i] = get_arg_val<uint32_t>(i + RANK + 3);
        dims_to_flip[i] = get_arg_val<uint32_t>(i + RANK + RANK + 3);
    }

    // Derived constants
    constexpr uint32_t FACE_HW = FACE_HEIGHT * FACE_WIDTH;
    constexpr uint32_t FACE_HW_BYTES = FACE_HW * element_size;
    constexpr uint32_t NUM_FACES_H = TILE_HEIGHT / FACE_HEIGHT;
    constexpr uint32_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;
    constexpr uint32_t NUM_FACES = NUM_FACES_H * NUM_FACES_W;
    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_WIDTH * element_size;
    const bool is_vertical_flip = static_cast<bool>(dims_to_flip[RANK - 2]);
    const bool is_horizontal_flip = static_cast<bool>(dims_to_flip[RANK - 1]);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const DataFormat data_format = get_dataformat(cb_id);
    const uint32_t tile_size = get_tile_size(cb_id);
    const auto s0 = TensorAccessor(src_args, src_addr, tile_size);

    // Face reading order depends on type of flip we performing
    static const uint32_t order_array[4][NUM_FACES] = {
        {0, 1, 2, 3},  // No flip
        {1, 0, 3, 2},  // Horizontal flip
        {2, 3, 0, 1},  // Vertical flip
        {3, 2, 1, 0}   // Both flips
    };

    // Select the appropriate face order based on flip flags
    uint32_t order_index = (is_horizontal_flip ? 1 : 0) + (is_vertical_flip ? 2 : 0);
    const uint32_t* face_reading_order = order_array[order_index];

    for (uint32_t tile_id = start_tile; tile_id < end_tile; ++tile_id) {
        cb_reserve_back(cb_id, onetile);
        uint32_t l1_buf_addr = get_write_ptr(cb_id);
        uint32_t save_addr = l1_buf_addr;  // save base address for debug print

        uint32_t src_tile_id = calc_src_tile_index(tile_id, RANK, dims_to_flip, tiled_shape, tile_strides);

        noc_async_read_tile(src_tile_id, s0, l1_buf_addr);
        noc_async_read_barrier();

        uint8_t* tile_ptr = reinterpret_cast<uint8_t*>(l1_buf_addr);

        for (uint32_t i = 0; i < NUM_FACES; i++) {
            uint32_t face_index = face_reading_order[i];
            uint8_t* face_ptr = tile_ptr + face_index * FACE_HW_BYTES;

            // if vertical flip then read rows in reverse order
            int32_t step = is_vertical_flip ? -1 : 1;
            int32_t start = is_vertical_flip ? FACE_HEIGHT - 1 : 0;
            int32_t end = is_vertical_flip ? -1 : FACE_HEIGHT;

            for (int32_t face_row = start; face_row != end; face_row += step) {
                uint8_t* row_bytes = face_ptr + face_row * SUBTILE_LINE_BYTES;

                if (is_horizontal_flip) {
                    // flip elements within the row
                    for (uint32_t i = 0; i < FACE_WIDTH / 2; ++i) {
                        uint32_t left = i * element_size;
                        uint32_t right = (FACE_WIDTH - 1 - i) * element_size;
                        for (uint32_t b = 0; b < element_size; ++b) {
                            uint8_t tmp = row_bytes[left + b];
                            row_bytes[left + b] = row_bytes[right + b];
                            row_bytes[right + b] = tmp;
                        }
                    }
                }
            }
        }

        cb_push_back(cb_id, onetile);
    }
}
