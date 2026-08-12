// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize writer — BRISC / NoC1.
//
// Raw-API justification (op_design.md §7.2): the only stick-writing helper,
// `dataflow_kernel_lib::write_sticks_after_untilize`, indexes its DESTINATION by
// row and copies `row_bytes` per row (tilize_helpers_dataflow.inl:232-236) — it
// writes STICKS. Our destination is a TILE tensor whose pages are whole tiles,
// so using it would scatter each tile's bytes across `tile_h` destination
// stick-pages. `kernel_lib` has no tile-page writer, so `noc_async_write` +
// `TensorAccessor::get_noc_addr(tile_index)` is the only mechanism.
//
// Transaction shape: whole-page (tile) coalesced writes (master.md B5), `w`
// writes issued back-to-back with ONE barrier per block (B7).
//
// `coalesce_writes == 0` (B5 off-arm) and `stub_write == 1` (the /perf-measure
// write ablation) are LEVER COUNTERFACTUALS. At their defaults (1, 0) the
// emitted code is the whole-page write loop and nothing else.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t nt_h = get_compile_time_arg_val(1);       // tile-rows
    constexpr uint32_t n_wchunks = get_compile_time_arg_val(2);  // column-blocks per tile-row
    constexpr uint32_t Wt = get_compile_time_arg_val(3);         // tile-columns (page stride)
    constexpr uint32_t wt_block = get_compile_time_arg_val(4);   // the block-width knob
    constexpr uint32_t wt_tail = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t coalesce_writes = get_compile_time_arg_val(7);  // lever B5 (1 = on)
    constexpr uint32_t stub_write = get_compile_time_arg_val(8);       // ablation (0 = off)
    constexpr auto dst_args = TensorAccessorArgs<9>();

    // B5 off-arm granularity: a tile is four faces, so the non-coalesced arm
    // issues one write per face instead of one per page.
    constexpr uint32_t FACES_PER_TILE = 4;
    constexpr uint32_t face_bytes = out_tile_bytes / FACES_PER_TILE;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1);
    const uint32_t nb = get_arg_val<uint32_t>(2);

    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t i = 0; i < nb; ++i) {
        const uint32_t b = b0 + i;
        const uint32_t wchunk = b / nt_h;      // column-block index
        const uint32_t r = b - wchunk * nt_h;  // global tile-row index
        const uint32_t w = (wchunk == n_wchunks - 1) ? wt_tail : wt_block;  // tiles this block
        const uint32_t c0 = wchunk * wt_block;                              // first tile-column

        cb_wait_front(cb_output_tiles, w);
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);
        const uint32_t first_page = r * Wt + c0;

        for (uint32_t t = 0; t < w; ++t) {
            if constexpr (stub_write == 0) {
                if constexpr (coalesce_writes == 1) {
                    noc_async_write(l1_addr, dst.get_noc_addr(first_page + t), out_tile_bytes);
                } else {
                    for (uint32_t f = 0; f < FACES_PER_TILE; ++f) {
                        noc_async_write(
                            l1_addr + f * face_bytes, dst.get_noc_addr(first_page + t, f * face_bytes), face_bytes);
                    }
                }
            }
            l1_addr += out_tile_bytes;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, w);
    }
}
