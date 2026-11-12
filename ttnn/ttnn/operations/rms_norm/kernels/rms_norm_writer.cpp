// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (regime R1). store_chunk per row-chunk, mirroring the reader's batching:
//   TILE: wait c tiles, <= c tile writes (pad columns >= core_w_tiles skipped), one barrier, pop c.
//   RM:   write_sticks_after_untilize for full-width chunks (valid sticks / valid bytes only);
//         the tail chunk of a K>1 row keeps the nominal c*32-element L1 stride (hand-written,
//         see tilize_helpers_dataflow.inl:208-209).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(8);

    constexpr uint32_t block_rows = get_compile_time_arg_val(9);
    constexpr uint32_t w_chunk_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t core_w_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(13);
    constexpr bool is_rm = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t W = get_compile_time_arg_val(15);
    constexpr uint32_t num_valid_sticks = get_compile_time_arg_val(16);
    constexpr uint32_t out_elem = get_compile_time_arg_val(17);
    constexpr auto out_args = TensorAccessorArgs<18>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(3);

    constexpr uint32_t chunk_elems = w_chunk_tiles * 32;
    const auto out_acc = TensorAccessor(out_args, out_addr);

    const uint32_t num_blocks = (num_tile_rows + block_rows - 1) / block_rows;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_row = block * block_rows;
        const uint32_t rows_this_block =
            (num_tile_rows - first_row < block_rows) ? (num_tile_rows - first_row) : block_rows;
        // Compute emits chunk-major (all rows of chunk k, then chunk k+1), so the writer walks the same order.
        for (uint32_t k = 0; k < num_w_chunks; ++k) {
            for (uint32_t i = 0; i < rows_this_block; ++i) {
                const uint32_t tile_row = start_tile_row + first_row + i;
                if constexpr (!is_rm) {
                    constexpr uint32_t out_tile_bytes = get_tile_size(cb_out);
                    cb_wait_front(cb_out, w_chunk_tiles);
                    uint32_t l1 = get_read_ptr(cb_out);
                    for (uint32_t j = 0; j < w_chunk_tiles; ++j) {
                        const uint32_t col = k * w_chunk_tiles + j;
                        if (col < core_w_tiles) {
                            noc_async_write_page(tile_row * tensor_w_tiles + w_tile_start + col, out_acc, l1);
                        }
                        l1 += out_tile_bytes;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, w_chunk_tiles);
                } else {
                    const uint32_t first_stick = tile_row * 32;
                    const uint32_t sticks =
                        (num_valid_sticks - first_stick < 32) ? (num_valid_sticks - first_stick) : 32;
                    const uint32_t elem_off = (w_tile_start * 32) + k * chunk_elems;
                    const uint32_t valid_elems = (W - elem_off < chunk_elems) ? (W - elem_off) : chunk_elems;
                    const uint32_t valid_bytes = valid_elems * out_elem;
                    if ((valid_elems + 31) / 32 == w_chunk_tiles) {
                        dataflow_kernel_lib::write_sticks_after_untilize<cb_out_rm>(
                            out_acc, sticks, valid_bytes, first_stick, elem_off * out_elem);
                    } else {
                        cb_wait_front(cb_out_rm, w_chunk_tiles);
                        uint32_t l1 = get_read_ptr(cb_out_rm);
                        for (uint32_t s = 0; s < sticks; ++s) {
                            noc_async_write(
                                l1, out_acc.get_noc_addr(first_stick + s, elem_off * out_elem), valid_bytes);
                            l1 += chunk_elems * out_elem;
                        }
                        noc_async_write_barrier();
                        cb_pop_front(cb_out_rm, w_chunk_tiles);
                    }
                }
            }
        }
    }
}
