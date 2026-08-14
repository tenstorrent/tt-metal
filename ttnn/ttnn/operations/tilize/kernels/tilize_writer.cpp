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

namespace {

// block index -> (tile-row, W chunk) for a W_BLOCKS core, matching the reader's
// W-chunk-major ordering. `nt_h` is a TEMPLATE parameter so the wide/short
// regime (`nt_h == 1`) folds the div/mod away entirely.
template <uint32_t nt_h>
struct BlockIndex {
    uint32_t row = 0;
    uint32_t wc = 0;
    uint32_t start_block_ = 0;

    FORCE_INLINE void init(uint32_t start_block) { start_block_ = start_block; }

    FORCE_INLINE void seek(uint32_t i) {
        const uint32_t b = start_block_ + i;
        row = b % nt_h;
        wc = b / nt_h;
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = 16;
    // Perf 1: ONE pre-stamped whole-pad output tile, writer-private (no
    // producer/consumer handshake — this kernel both fills it and sources from
    // it). Allocated by the host only when `pad_scratch` is 1.
    constexpr uint32_t cb_pad_scratch = 2;
    constexpr uint32_t TILE_W = 32;  // a tile is always 32 wide (hardware fact, not a knob)

    constexpr uint32_t placement = get_compile_time_arg_val(0);  // P_ACCESSOR / P_LOCAL_SHARD
    constexpr uint32_t work_mode = get_compile_time_arg_val(1);  // W_BLOCKS / W_REGION
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(2);   // the W block factor
    constexpr uint32_t nt_h = get_compile_time_arg_val(3);
    constexpr uint32_t wt = get_compile_time_arg_val(4);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(5);  // W chunks per shard row (W_REGION)
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(6);
    // master.md B8, WRITE side — the twin of the reader's issue-ahead window.
    // 1 = block i's writes are issued BEFORE block i-1's barrier, so a write is
    // always in flight across the block boundary; 0 = barrier then issue, which
    // drains the write NoC at every block. The split-DM ablation (Refinement 3)
    // put the WRITE half on the critical path — it is the slower of the two on
    // every real-work regime (a: 59.5 vs 43.9 us) — which is why this twin
    // exists at all. The host only sets it when the output CB is EXACTLY two
    // blocks deep. Every write is one whole tile PAGE (master.md B5).
    constexpr uint32_t write_trid = get_compile_time_arg_val(7);
    // --- Refinement 4: the OUTPUT-format pad stamp ---------------------------
    // 1 = re-stamp each finished tile's pad region with `pad_word_out`, which the
    // host packed in the OUTPUT element format. Enabled ONLY when the fill cannot
    // survive the input format's round-trip (a widening cast with a fill that is
    // inexact in the input dtype) — otherwise the reader's input-format fill is
    // already exact and this compiles away entirely.
    constexpr uint32_t out_fill = get_compile_time_arg_val(8);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t tile_h = get_compile_time_arg_val(10);
    // The data extent inside the PADDED target — the same four quantities the
    // reader uses to decide what to fill, so the two fills cannot disagree.
    constexpr uint32_t h_in = get_compile_time_arg_val(11);
    constexpr uint32_t w_in_elems = get_compile_time_arg_val(12);
    constexpr uint32_t nth_per_img = get_compile_time_arg_val(13);
    constexpr uint32_t n_img_in = get_compile_time_arg_val(14);
    // --- Perf 1: whole-pad tiles come from a pre-stamped scratch tile ---------
    // 1 = the padded target contains at least one WHOLE pad tile (every element of
    // it is pad) AND `cb_pad_scratch` exists. The host owns this predicate (see
    // `pad_scratch` in tilize_program_descriptor.py) so the CB allocation and this
    // branch cannot disagree; 0 compiles the whole mechanism away, which is what
    // keeps a ragged-tail-only geometry byte-identical to before.
    constexpr uint32_t pad_scratch = get_compile_time_arg_val(15);
    constexpr auto dst_args = TensorAccessorArgs<16>();

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
    // stamp's geometry. On W_BLOCKS the pair comes from `idx`, which every loop
    // below `seek`s at the top of its body.
    BlockIndex<nt_h> idx;
    idx.init(start_block);

    auto tile_row_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1 /* W_REGION */) {
            return tile_row0 + i / n_chunks;
        } else {
            return idx.row;
        }
    };
    auto tile_col_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1 /* W_REGION */) {
            const uint32_t r = i / n_chunks;
            return tile_col0 + (i - r * n_chunks) * wt_chunk;
        } else {
            return idx.wc * wt_chunk;  // W-chunk-major ordering
        }
    };
    auto first_page_of = [&](uint32_t i) -> uint32_t { return tile_row_of(i) * wt + tile_col_of(i); };

    // The pad geometry of one tile — the SINGLE source for both the stamp and the
    // whole-pad-tile test below.
    auto valid_rows_of = [&](uint32_t i) -> uint32_t {
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
        return valid_rows;
    };
    auto valid_cols_of = [&](uint32_t i, uint32_t k) -> uint32_t {
        const uint32_t col0 = (tile_col_of(i) + k) * TILE_W;
        uint32_t valid_cols = 0;
        if (col0 < w_in_elems) {
            valid_cols = w_in_elems - col0;
            if (valid_cols > TILE_W) {
                valid_cols = TILE_W;
            }
        }
        return valid_cols;
    };

    // ── Perf 1: the pre-stamped WHOLE-pad tile ───────────────────────────────
    // A whole pad tile — one whose every element is the fill — used to cost
    // ~tile_h*32 rv32 volatile L1 stores, EVERY block, and that made the padded
    // widening cast the op's most expensive path by a wide margin: `writer_stamp`
    // measured 172,898 of the 386,280 ns wall on
    // [1,1,1024,2048] bf16 -> fp32 -> [1,1,2048,2048] (45%, the single largest
    // attributable stage), and a whole-op simultaneous ablation left 94% of the
    // wall standing with nothing but the CB handshake and the stamp.
    //
    // Every such tile is byte-identical, so it is built ONCE per core into
    // `cb_pad_scratch` and then reused two ways:
    //   * P_ACCESSOR — the outgoing page write is issued straight FROM the scratch
    //     tile (`src_of` below). The CB slot is never touched at all: not one
    //     store, not one byte moved.
    //   * P_LOCAL_SHARD — there is no outgoing write (the CB *is* the tensor), so
    //     the scratch tile is copied into the slot with ONE local L1->L1 transfer.
    //
    // Measured whole-op: 386,749 -> 142,761 ns (2.71x) on the focus case,
    // 366,475 -> 22,487 ns (16.3x) on a padded HEIGHT-sharded output, and FLAT on
    // geometries with only ragged tails.
    //
    // The fill is LAZY on purpose. Stamping the scratch at kernel start cost
    // +4.3 us on a 6.7 us [1,1,50,50]->[1,1,64,64] — a pure regression on every
    // geometry that has tails but no whole pad tile. `pad_scratch` (host-derived)
    // additionally deletes the mechanism outright where the target cannot contain
    // one, so those cells keep the old code byte-for-byte.
    //
    // RAW-DATAFLOW NOTE: the L1->L1 replicate is a local `noc_async_read` from
    // `get_noc_addr(scratch)`, not a `noc_async_write`. The WRITE command buffer
    // already carries the outgoing page writes and, on the B8 path, a trid'd write
    // that must stay in flight across the block boundary; issuing the copy on the
    // idle READ command buffer gives it an INDEPENDENT `noc_async_read_barrier()`
    // that orders it against the page write without touching the write-trid state.
    constexpr bool use_scratch = out_fill && pad_scratch;
    uint32_t scratch_addr = 0;
    uint64_t scratch_noc = 0;
    bool scratch_ready = false;
    if constexpr (use_scratch) {
        scratch_addr = get_write_ptr(cb_pad_scratch);
        scratch_noc = get_noc_addr(scratch_addr);
    }
    auto ensure_scratch = [&]() {
        if constexpr (use_scratch) {
            if (!scratch_ready) {
                tilize_kernels::fill_l1_with_val<out_elem_bytes>(
                    scratch_addr, tile_h * TILE_W * out_elem_bytes, pad_word_out);
                scratch_ready = true;
            }
        }
    };

    // Stamp the pad region of this block's wt_chunk tiles, in the OUTPUT element
    // format. Compiles to nothing when `out_fill` is 0 (every path but a padded
    // widening cast).
    auto stamp_pad = [&](uint32_t i, uint32_t l1_addr) {
        if constexpr (out_fill) {
            const uint32_t valid_rows = valid_rows_of(i);
            bool copied = false;
            for (uint32_t k = 0; k < wt_chunk; ++k) {
                const uint32_t valid_cols = valid_cols_of(i, k);
                const uint32_t addr = l1_addr + k * out_tile_bytes;
                const bool whole_pad_tile = (valid_rows == 0 || valid_cols == 0);
                if constexpr (use_scratch) {
                    if (whole_pad_tile) {
                        ensure_scratch();
                        if constexpr (placement == 1 /* P_LOCAL_SHARD */) {
                            // No outgoing write to re-source: replicate into the slot.
                            noc_async_read(scratch_noc, addr, out_tile_bytes);
                            copied = true;
                        }
                        // P_ACCESSOR: nothing to do here at all — `src_of` sends the
                        // page from the scratch tile and the slot is never read.
                        continue;
                    }
                }
                // A ragged W/H tail (or a geometry with no whole pad tile at all):
                // stamped in place, exactly as before.
                tilize_kernels::fill_tile_pad<tile_h, TILE_W, out_elem_bytes>(
                    addr, valid_rows, valid_cols, pad_word_out);
            }
            if (copied) {
                // The replicate must LAND before anything reads the slot.
                noc_async_read_barrier();
            }
        }
    };

    // The L1 address an outgoing page write is sourced from: the pre-stamped
    // scratch tile for a whole pad tile, the CB slot for everything else.
    auto src_of = [&](uint32_t i, uint32_t k, uint32_t l1_addr) -> uint32_t {
        if constexpr (use_scratch && placement != 1) {
            if (valid_rows_of(i) == 0 || valid_cols_of(i, k) == 0) {
                return scratch_addr;
            }
        }
        return l1_addr;
    };

    if constexpr (placement == 1 /* P_LOCAL_SHARD */) {
        // ── ZERO-COPY ────────────────────────────────────────────────────
        // Compute packed straight into the resident output shard. Drain only —
        // no NoC write, and the CB keeps exactly one consumer. The pad stamp still
        // runs: the CB *is* the output tensor, so it edits the shard in place.
        for (uint32_t i = 0; i < num_blocks; ++i) {
            idx.seek(i);
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
            idx.seek(i);
            const uint32_t first_page = first_page_of(i);
            // The still-unbarriered in-flight block AND this one.
            cb_wait_front(cb_output_tiles, in_flight ? 2 * wt_chunk : wt_chunk);
            uint32_t l1_addr = slot_base + slot * slot_bytes;

            stamp_pad(i, l1_addr);  // before the bytes leave L1
            noc_async_write_set_trid(trid_issue);
            for (uint32_t k = 0; k < wt_chunk; ++k) {
                noc_async_write(src_of(i, k, l1_addr), accessor.get_noc_addr(first_page + k), out_tile_bytes);
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
        idx.seek(i);
        const uint32_t first_page = first_page_of(i);

        cb_wait_front(cb_output_tiles, wt_chunk);
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);

        stamp_pad(i, l1_addr);  // before the bytes leave L1

        // One whole tile PAGE per transaction (master.md B5) ...
        for (uint32_t k = 0; k < wt_chunk; ++k) {
            noc_async_write(src_of(i, k, l1_addr), accessor.get_noc_addr(first_page + k), out_tile_bytes);
            l1_addr += out_tile_bytes;
        }

        // ... and ONE barrier per block (master.md B7).
        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, wt_chunk);
    }
}
