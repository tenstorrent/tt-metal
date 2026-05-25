// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for iterative_topk with TILE layout input.
// Reads tile data from the input CB (face-aware addressing), performs iterative
// max k times per logical row, and writes ROW_MAJOR output pages to DRAM.
// Output indices are UINT16.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_ELEMENTS = 1024;  // 32 * 32
constexpr uint32_t FACE_SIZE = 256;       // 16 * 16

union FloatUint32 {
    float f;
    uint32_t u;
};

// Read a float element from tile data given the intra-tile row and global column.
// tiles_base points to width_tiles contiguous tiles in L1.
FORCE_INLINE float read_tile_element_f(volatile tt_l1_ptr uint32_t* tiles_base, uint32_t intra_tile_row, uint32_t col) {
    uint32_t tile_col_idx = col >> 5;  // col / 32
    uint32_t intra_col = col & 31;     // col % 32
    uint32_t face = ((intra_tile_row >> 4) << 1) + (intra_col >> 4);
    uint32_t face_row = intra_tile_row & 15;
    uint32_t face_col = intra_col & 15;
    uint32_t offset = tile_col_idx * TILE_ELEMENTS + face * FACE_SIZE + face_row * 16 + face_col;
    FloatUint32 conv;
    conv.u = tiles_base[offset];
    return conv.f;
}

// Read a uint32 element from tile data (for bit manipulation / comparison).
FORCE_INLINE uint32_t
read_tile_element_u32(volatile tt_l1_ptr uint32_t* tiles_base, uint32_t intra_tile_row, uint32_t col) {
    uint32_t tile_col_idx = col >> 5;
    uint32_t intra_col = col & 31;
    uint32_t face = ((intra_tile_row >> 4) << 1) + (intra_col >> 4);
    uint32_t face_row = intra_tile_row & 15;
    uint32_t face_col = intra_col & 15;
    uint32_t offset = tile_col_idx * TILE_ELEMENTS + face * FACE_SIZE + face_row * 16 + face_col;
    return tiles_base[offset];
}

// Write a uint32 value to a tile element position.
FORCE_INLINE void write_tile_element_u32(
    volatile tt_l1_ptr uint32_t* tiles_base, uint32_t intra_tile_row, uint32_t col, uint32_t value) {
    uint32_t tile_col_idx = col >> 5;
    uint32_t intra_col = col & 31;
    uint32_t face = ((intra_tile_row >> 4) << 1) + (intra_col >> 4);
    uint32_t face_row = intra_tile_row & 15;
    uint32_t face_col = intra_col & 15;
    uint32_t offset = tile_col_idx * TILE_ELEMENTS + face * FACE_SIZE + face_row * 16 + face_col;
    tiles_base[offset] = value;
}

void kernel_main() {
    const uint32_t output_values_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);
    const uint32_t total_rows = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_values = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_indices = get_compile_time_arg_val(2);
    constexpr uint32_t width = get_compile_time_arg_val(3);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t k = get_compile_time_arg_val(5);
    constexpr uint32_t input_tile_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t output_values_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t output_indices_page_size = get_compile_time_arg_val(8);

    constexpr uint32_t values_accessor_offset = 9;
    constexpr auto values_args = TensorAccessorArgs<values_accessor_offset>();
    const auto values_accessor = TensorAccessor(values_args, output_values_addr);

    constexpr uint32_t indices_accessor_offset = values_args.next_compile_time_args_offset();
    constexpr auto indices_args = TensorAccessorArgs<indices_accessor_offset>();
    const auto indices_accessor = TensorAccessor(indices_args, output_indices_addr);

    constexpr uint32_t NEG_INF_U32 = 0xFF800000u;

    // Reserve output CB space once — reused as scratch for each output row
    cb_reserve_back(cb_out_values, 1);
    cb_reserve_back(cb_out_indices, 1);
    uint32_t out_values_l1 = get_write_ptr(cb_out_values);
    uint32_t out_indices_l1 = get_write_ptr(cb_out_indices);

    volatile tt_l1_ptr uint32_t* out_vals_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_values_l1);
    volatile tt_l1_ptr uint16_t* out_idxs = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_indices_l1);

    for (uint32_t ht = start_height_tile; ht < end_height_tile; ht++) {
        // Wait for all width_tiles pages (one full height-tile row of input)
        cb_wait_front(cb_input, width_tiles);
        volatile tt_l1_ptr uint32_t* tiles_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_input));

        for (uint32_t intra_row = 0; intra_row < 32; intra_row++) {
            uint32_t global_row = ht * 32 + intra_row;
            if (global_row >= total_rows) {
                break;
            }

            // Iterative max: find top-k by repeated scanning
            for (uint32_t ki = 0; ki < k; ki++) {
                float max_val = read_tile_element_f(tiles_base, intra_row, 0);
                uint32_t max_idx = 0;

                for (uint32_t col = 1; col < width; col++) {
                    float val = read_tile_element_f(tiles_base, intra_row, col);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                out_vals_u32[ki] = read_tile_element_u32(tiles_base, intra_row, max_idx);
                out_idxs[ki] = (uint16_t)max_idx;

                // Mask found max with -inf so next iteration skips it
                write_tile_element_u32(tiles_base, intra_row, max_idx, NEG_INF_U32);
            }

            // Write output row to DRAM
            noc_async_write_page(global_row, values_accessor, out_values_l1);
            noc_async_write_page(global_row, indices_accessor, out_indices_l1);
            noc_async_write_barrier();
        }

        cb_pop_front(cb_input, width_tiles);
    }
}
