// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader (NCRISC / NoC0) — block operations `read_x_row_block` and `replicate_row_block`.
//
// Per block (image group ig, row group rg, column group cg):
//   read_x_row_block:    reserve block_pages = block_images * block_tile_cols CB pages (nominal),
//                        one partial DRAM read (x_row_read_bytes: face-0 + face-1 row 0) per x tile
//                        into the start of its CB page, ONE read barrier per block.
//   replicate_row_block: per tile, row 0 of faces 0/1 -> every other row of faces 0..3 in place via
//                        self-aimed single-packet L1->L1 NoC reads (local_copy_helpers_dataflow),
//                        one barrier per tile, push 1 page per tile so the writer starts early.
//   ragged block:        the unused (block_pages - real) pages are pushed at block end.
// Byte moves only: bit-exact for NaN payloads, signed zeros, denormals. Rows 1..31 of x (tile
// padding) never reach a destination row: every row of every face is a source or overwritten.
//
// Raw NoC page read for x: no kernel_lib helper moves a partial interleaved DRAM page (design,
// "Helpers considered and rejected").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

template <uint32_t face_row_bytes, uint32_t face_bytes, uint32_t face_rows>
FORCE_INLINE void replicate_row_in_tile(Noc noc, uint32_t tile_l1) {
    // Two halves: left (faces 0, 2) sourced from face-0 row 0, right (faces 1, 3) from face-1 row 0.
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t src = tile_l1 + half * face_bytes;
        dataflow_kernel_lib::set_read_state<face_row_bytes>(noc, src);
        // top face (0 or 1): rows 1..face_rows-1
        for (uint32_t r = 1; r < face_rows; ++r) {
            dataflow_kernel_lib::read_with_state<face_row_bytes>(noc, src + r * face_row_bytes, src);
        }
        // bottom face (2 or 3): rows 0..face_rows-1
        const uint32_t bottom = tile_l1 + (2 + half) * face_bytes;
        for (uint32_t r = 0; r < face_rows; ++r) {
            dataflow_kernel_lib::read_with_state<face_row_bytes>(noc, bottom + r * face_row_bytes, src);
        }
    }
}

void kernel_main() {
    constexpr uint32_t cb_bcast_row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t x_row_read_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t face_rows = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();

    uint32_t arg = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t first_block = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_tile_cols = get_arg_val<uint32_t>(arg++);
    const uint32_t row_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t col_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t block_tile_cols = get_arg_val<uint32_t>(arg++);
    const uint32_t block_images = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_images = get_arg_val<uint32_t>(arg++);

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    Noc noc;

    const uint32_t block_pages = block_images * block_tile_cols;

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t b = first_block + block_idx;
        const uint32_t cg = b % col_groups;
        const uint32_t ig = b / (col_groups * row_groups);  // rg = (b / col_groups) % row_groups: unused here
        const uint32_t n0 = ig * block_images;
        const uint32_t c0 = cg * block_tile_cols;
        const uint32_t this_images = (tensor_images - n0) < block_images ? (tensor_images - n0) : block_images;
        const uint32_t this_cols =
            (tensor_tile_cols - c0) < block_tile_cols ? (tensor_tile_cols - c0) : block_tile_cols;
        const uint32_t real_pages = this_images * this_cols;

        cb_reserve_back(cb_bcast_row_tiles, block_pages);
        const uint32_t block_l1 = get_write_ptr(cb_bcast_row_tiles);

        {
            DeviceZoneScopedN("read_x_row");
            uint32_t page_l1 = block_l1;
            for (uint32_t i = 0; i < this_images; ++i) {
                const uint32_t x_page_base = (n0 + i) * tensor_tile_cols + c0;
                for (uint32_t c = 0; c < this_cols; ++c) {
                    noc_async_read(x_accessor.get_noc_addr(x_page_base + c), page_l1, x_row_read_bytes);
                    page_l1 += tile_bytes;
                }
            }
            noc_async_read_barrier();
        }

        uint32_t page_l1 = block_l1;
        for (uint32_t k = 0; k < real_pages; ++k) {
            {
                DeviceZoneScopedN("replicate_row");
                replicate_row_in_tile<face_row_bytes, face_bytes, face_rows>(noc, page_l1);
                noc_async_read_barrier();
            }
            cb_push_back(cb_bcast_row_tiles, 1);
            page_l1 += tile_bytes;
        }
        if (real_pages < block_pages) {
            cb_push_back(cb_bcast_row_tiles, block_pages - real_pages);
        }
    }
}
