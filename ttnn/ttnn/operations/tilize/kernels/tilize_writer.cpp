// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize writer (BRISC / NOC1).
//
// Drains cb_output_tiles one block at a time (1 tile-row x WT_CHUNK
// tile-columns) and writes each tile as ONE whole page (master.md B5), with ONE
// barrier per block (master.md B7).
//
// Two PLACEMENT regimes (op_design.md §5.2):
//
//   P_ACCESSOR    — TensorAccessor over the interleaved (or non-local sharded)
//                   destination. Issues the writes described above.
//   P_LOCAL_SHARD — cb_output_tiles is ALIASED on this core's resident TILE
//                   shard, so compute already packed straight into the output
//                   tensor. The writer issues NO NoC write: it only DRAINS the
//                   CB, which is kept precisely so the CB still has exactly one
//                   consumer (op_design.md §6).
//
// ... and two work assignments (W_BLOCKS = a range of the global W-chunk-major
// block index; W_REGION = this core's own shard tile region, tile-row-major with
// the W chunk innermost).
//
// HELPER SUBSTITUTION note: the only kernel_lib writer for this family is
// dataflow_kernel_lib::write_sticks_after_untilize, which is the INVERSE
// direction — its contract (tilize_helpers_dataflow.hpp:98-102) de-interleaves
// tiles into row-major STICKS. Our destination pages are whole TILE pages that
// need no de-interleave; using it would write stick fragments into a tiled
// buffer. No kernel_lib helper covers CB-tiles -> tiled-tensor pages, so raw
// TensorAccessor + noc_async_write is the correct mechanism here.

#include "api/dataflow/dataflow_api.h"
// fill_tile_pad / fill_l1_with_val — shared with the reader (which fills the same
// pad region in the INPUT element format on the way in). See the header for why
// the fill is materialized twice.
#include "ttnn/ttnn/operations/tilize/kernels/tilize_fill.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = 16;
    constexpr uint32_t TILE_W = 32;  // a tile is always 32 wide (hardware fact, not a knob)

    constexpr uint32_t placement = get_compile_time_arg_val(0);  // P_ACCESSOR / P_LOCAL_SHARD
    constexpr uint32_t work_mode = get_compile_time_arg_val(1);  // W_BLOCKS / W_REGION
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(2);   // the W block factor
    constexpr uint32_t nt_h = get_compile_time_arg_val(3);
    constexpr uint32_t wt = get_compile_time_arg_val(4);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(5);  // W chunks per shard row (W_REGION)
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(6);
    // Lever `block_write` (master.md B7): 1 = one barrier per BLOCK (optimal),
    // 0 = one barrier per tile page (the counterfactual OFF arm the bench measures).
    constexpr uint32_t block_write = get_compile_time_arg_val(7);
    // Classification ablation (op_design.md §9.1): drop the NoC payload, keep the
    // CB handshake, barriers and loop trip counts. Always 0 in production.
    constexpr uint32_t ablate_dm = get_compile_time_arg_val(8);
    // Lever `page_write` (master.md B5): 1 = one whole tile PAGE per transaction
    // (optimal), 0 = two half-page transactions (the sub-page-scatter OFF arm).
    constexpr uint32_t page_write = get_compile_time_arg_val(9);
    // Lever `write_trid` (master.md B8, WRITE side — the twin of the reader's
    // read_trid). 1 = block i's writes are issued BEFORE block i-1's barrier, so
    // a write is always in flight across the block boundary; 0 = barrier then
    // issue, which drains the write NoC at every block. The split-DM ablation
    // (Refinement 3) put the WRITE half on the critical path — it is the slower
    // of the two on every real-work regime (a: 59.5 vs 43.9 us) — which is why
    // this twin exists at all. The host only sets it when the output CB is
    // EXACTLY two blocks deep and every write is a whole page.
    constexpr uint32_t write_trid = get_compile_time_arg_val(10);
    // --- Refinement 4: the OUTPUT-format pad stamp ---------------------------
    // 1 = re-stamp each finished tile's pad region with `pad_word_out`, which the
    // host packed in the OUTPUT element format. Enabled ONLY when the fill cannot
    // survive the input format's round-trip (a widening cast with a fill that is
    // inexact in the input dtype) — otherwise the reader's input-format fill is
    // already exact and this compiles away entirely.
    constexpr uint32_t out_fill = get_compile_time_arg_val(11);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t tile_h = get_compile_time_arg_val(13);
    // The data extent inside the PADDED target — the same four quantities the
    // reader uses to decide what to fill, so the two fills cannot disagree.
    constexpr uint32_t h_in = get_compile_time_arg_val(14);
    constexpr uint32_t w_in_elems = get_compile_time_arg_val(15);
    constexpr uint32_t nth_per_img = get_compile_time_arg_val(16);
    constexpr uint32_t n_img_in = get_compile_time_arg_val(17);
    constexpr auto dst_args = TensorAccessorArgs<18>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t tile_row0 = get_arg_val<uint32_t>(3);  // W_REGION: region origin
    const uint32_t tile_col0 = get_arg_val<uint32_t>(4);
    const uint32_t pad_word_out = get_arg_val<uint32_t>(5);  // fill, in the OUTPUT format

    if (num_blocks == 0) {
        return;
    }

    // block index -> (tile-row, first tile-column) of the block, per work
    // assignment. THE single source for both the destination page ids and the pad
    // stamp's geometry.
    auto tile_row_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1 /* W_REGION */) {
            return tile_row0 + i / n_chunks;
        } else {
            return (start_block + i) % nt_h;
        }
    };
    auto tile_col_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1 /* W_REGION */) {
            const uint32_t r = i / n_chunks;
            return tile_col0 + (i - r * n_chunks) * wt_chunk;
        } else {
            return ((start_block + i) / nt_h) * wt_chunk;  // W-chunk-major ordering
        }
    };
    auto first_page_of = [&](uint32_t i) -> uint32_t { return tile_row_of(i) * wt + tile_col_of(i); };

    // Stamp the pad region of this block's wt_chunk tiles, in the OUTPUT element
    // format. Compiles to nothing when `out_fill` is 0 (every path but a padded
    // widening cast).
    auto stamp_pad = [&](uint32_t i, uint32_t l1_addr) {
        if constexpr (out_fill) {
            const uint32_t tile_row = tile_row_of(i);
            const uint32_t img = tile_row / nth_per_img;
            const uint32_t row_in_img = (tile_row % nth_per_img) * tile_h;
            // Rows of this tile-row that carry real data (0 for a whole pad tile-row).
            uint32_t valid_rows = 0;
            if (img < n_img_in && row_in_img < h_in) {
                valid_rows = h_in - row_in_img;
                if (valid_rows > tile_h) {
                    valid_rows = tile_h;
                }
            }
            const uint32_t tile_col = tile_col_of(i);
            for (uint32_t k = 0; k < wt_chunk; ++k) {
                const uint32_t col0 = (tile_col + k) * TILE_W;
                uint32_t valid_cols = 0;
                if (col0 < w_in_elems) {
                    valid_cols = w_in_elems - col0;
                    if (valid_cols > TILE_W) {
                        valid_cols = TILE_W;
                    }
                }
                tilize_kernels::fill_tile_pad<tile_h, TILE_W, out_elem_bytes>(
                    l1_addr + k * out_tile_bytes, valid_rows, valid_cols, pad_word_out);
            }
        }
    };

    if constexpr (placement == 1 /* P_LOCAL_SHARD */) {
        // ── ZERO-COPY ────────────────────────────────────────────────────
        // Compute packed straight into the resident output shard. Drain only —
        // no NoC write, and the CB keeps exactly one consumer. The pad stamp still
        // runs: the CB *is* the output tensor, so it edits the shard in place.
        for (uint32_t i = 0; i < num_blocks; ++i) {
            cb_wait_front(cb_output_tiles, wt_chunk);
            stamp_pad(i, get_read_ptr(cb_output_tiles));
            cb_pop_front(cb_output_tiles, wt_chunk);
        }
        return;
    }

    const auto accessor = TensorAccessor(dst_args, dst_addr);

    if constexpr (write_trid) {
        // ── B8 WRITE-side double-issue ───────────────────────────────────
        // The output CB is exactly two blocks deep (host precondition), so the
        // read pointer alternates between two fixed slots and no wrap
        // arithmetic is needed; cb_wait_front still provides all flow control.
        // noc_async_write_set_trid tags the write command buffer, and plain
        // noc_async_write leaves NOC_PACKET_TAG alone — so whole TILE pages
        // (> NOC_MAX_BURST_SIZE, hence not expressible with the one-packet
        // *_with_trid API) still carry the id.
        constexpr uint32_t slot_bytes = wt_chunk * out_tile_bytes;
        const uint32_t slot_base = get_read_ptr(cb_output_tiles);
        uint32_t slot = 0, trid_issue = 1, trid_wait = 1;
        bool in_flight = false;

        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t first_page = first_page_of(i);
            // The still-unbarriered in-flight block AND this one.
            cb_wait_front(cb_output_tiles, in_flight ? 2 * wt_chunk : wt_chunk);
            uint32_t l1_addr = slot_base + slot * slot_bytes;

            stamp_pad(i, l1_addr);  // before the bytes leave L1
            noc_async_write_set_trid(trid_issue);
            for (uint32_t k = 0; k < wt_chunk; ++k) {
                noc_async_write(l1_addr, accessor.get_noc_addr(first_page + k), out_tile_bytes);
                l1_addr += out_tile_bytes;
            }
            slot ^= 1;
            trid_issue ^= 3;  // alternate 1 <-> 2

            if (in_flight) {
                noc_async_write_barrier_with_trid(trid_wait);
                cb_pop_front(cb_output_tiles, wt_chunk);
                trid_wait ^= 3;
            }
            in_flight = true;
        }
        noc_async_write_barrier_with_trid(trid_wait);  // drain the last block
        cb_pop_front(cb_output_tiles, wt_chunk);
        // MANDATORY: put the command buffer's packet tag back to 0 before the
        // kernel exits. brisck.cc:91 asserts `ncrisc_noc_packet_tags_cleared`
        // (it reads NOC_PACKET_TAG on the WR / WR_REG / AT command buffers), so
        // a left-behind trid halts the core in firmware AFTER kernel_main
        // returns — which presents as a whole-grid hang at waypoint NKFW, not
        // as a CB deadlock. (The read cmd buf is not in that check, which is why
        // only the write side trips it.)
        noc_async_write_set_trid(0);
        return;
    }

    for (uint32_t i = 0; i < num_blocks; ++i) {
        const uint32_t first_page = first_page_of(i);

        cb_wait_front(cb_output_tiles, wt_chunk);
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);

        stamp_pad(i, l1_addr);  // before the bytes leave L1

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
