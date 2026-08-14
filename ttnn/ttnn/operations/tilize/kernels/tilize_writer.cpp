// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize writer (BRISC / NOC1).
//
// Drains cb_output_tiles one block at a time (1 tile-row x WT_CHUNK
// tile-columns) and writes each tile as ONE whole page (master.md B5), with ONE
// barrier per block (master.md B7).
//
// HELPER SUBSTITUTION note: the only kernel_lib writer for this family is
// dataflow_kernel_lib::write_sticks_after_untilize, which is the INVERSE
// direction — its contract (tilize_helpers_dataflow.hpp:98-102) de-interleaves
// tiles into row-major STICKS. Our destination pages are whole TILE pages that
// need no de-interleave; using it would write stick fragments into a tiled
// buffer. No kernel_lib helper covers CB-tiles -> tiled-tensor pages, so raw
// TensorAccessor + noc_async_write is the correct mechanism here.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = 16;

    constexpr uint32_t wt_chunk = get_compile_time_arg_val(0);  // the W block factor
    constexpr uint32_t nt_h = get_compile_time_arg_val(1);
    constexpr uint32_t wt = get_compile_time_arg_val(2);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(3);
    // Lever `block_write` (master.md B7): 1 = one barrier per BLOCK (optimal),
    // 0 = one barrier per tile page (the counterfactual OFF arm the bench measures).
    constexpr uint32_t block_write = get_compile_time_arg_val(4);
    // Classification ablation (op_design.md §9.1): drop the NoC payload, keep the
    // CB handshake, barriers and loop trip counts. Always 0 in production.
    constexpr uint32_t ablate_dm = get_compile_time_arg_val(5);
    // Lever `page_write` (master.md B5): 1 = one whole tile PAGE per transaction
    // (optimal), 0 = two half-page transactions (the sub-page-scatter OFF arm).
    constexpr uint32_t page_write = get_compile_time_arg_val(6);
    constexpr auto dst_args = TensorAccessorArgs<7>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    if (num_blocks == 0) {
        return;
    }

    const auto accessor = TensorAccessor(dst_args, dst_addr);

    for (uint32_t i = 0; i < num_blocks; ++i) {
        const uint32_t block = start_block + i;
        const uint32_t wc = block / nt_h;   // W-chunk-major block ordering
        const uint32_t row = block % nt_h;  // tile-row this block produces
        const uint32_t first_page = row * wt + wc * wt_chunk;

        cb_wait_front(cb_output_tiles, wt_chunk);
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);

        for (uint32_t k = 0; k < wt_chunk; ++k) {
            if constexpr (!ablate_dm) {
                if constexpr (page_write) {
                    noc_async_write(l1_addr, accessor.get_noc_addr(first_page + k), out_tile_bytes);
                } else {
                    // OFF arm: the same bytes split into two sub-page transactions.
                    constexpr uint32_t half = out_tile_bytes / 2;
                    noc_async_write(l1_addr, accessor.get_noc_addr(first_page + k), half);
                    noc_async_write(l1_addr + half, accessor.get_noc_addr(first_page + k, half), out_tile_bytes - half);
                }
            }
            l1_addr += out_tile_bytes;
            if constexpr (!block_write) {
                noc_async_write_barrier();  // OFF arm: barrier per transaction
            }
        }

        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, wt_chunk);
    }
}
