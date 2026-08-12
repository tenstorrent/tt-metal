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
// `resident == 1` (Refinement 2, A3/C14) is the SECOND production path: the
// output shard is this core's own L1 and `cb_output_tiles` is aliased onto it,
// so compute already packed each tile into its final home and the writer only
// drains the CB — zero NoC writes.
//
// `coalesce_writes == 0` (B5 off-arm) and `stub_write == 1` (the /perf-measure
// write ablation) are LEVER COUNTERFACTUALS. At their defaults (1, 0) the
// emitted code is the whole-page write loop and nothing else.

#include "api/dataflow/dataflow_api.h"
#include "pad_fill.hpp"  // the shared pad-fill store loop (also used by the reader)

// R7 — the OUTPUT-FORMAT PAD REWRITE (`out_pad_fix == 1`).
//
// The pad fill is written by the reader into the INPUT circular buffer, so it
// is necessarily packed in the INPUT element format (op_design.md §8.3). On a
// WIDENING cast that rounds the caller's number before the output ever sees it:
// `tilize(bf16_tensor, dtype=float32, pad_value=10.2)` came back with 10.1875 in
// the pad region (bf16's nearest), while the oracle — and the caller — expect
// fp32 10.2. Measured delta 0.0125, i.e. exactly one bf16 ulp at that value.
//
// The real data is unaffected (it IS input-format data; widening it is exact),
// so only the pad POSITIONS need rewriting, and only in the output format. The
// host arms this exclusively when the output format holds the fill strictly
// better than the input's does — so a same-dtype call, an integer call, a
// narrowing cast and any exactly-representable fill (0, 10.0, 42.0, 3.5, -18.0)
// all emit none of this.
//
// Positions, in TILE layout: a tile is four 16x16 faces, so datum (row, col)
// lives at face `(row/16)*2 + (col/16)`, index `(row%16)*16 + (col%16)` inside
// it. The pad region of a tile is "rows >= real_rows" (H tail and whole pad
// tile-ROWS) plus "cols >= real_cols" of the real rows (W tail and whole pad
// tile-COLUMNS), which is the SAME two-scalar split the reader computes — and
// each (row, face-half) piece is one contiguous run, so the store loop is the
// reader's `fill_pad_region`, not a per-datum walk.
FORCE_INLINE void fill_tile_pad(
    uint32_t tile_base, uint32_t real_rows, uint32_t real_cols, uint32_t tile_h, uint32_t out_elem, uint32_t pad_word) {
    constexpr uint32_t FACE_DIM = 16;
    for (uint32_t row = 0; row < tile_h; ++row) {
        const uint32_t from_col = (row >= real_rows) ? 0 : real_cols;
        if (from_col >= 2 * FACE_DIM) {
            continue;  // this row is entirely real
        }
        const uint32_t face_pair = (row / FACE_DIM) * 2;
        const uint32_t in_face_row = row % FACE_DIM;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t col0 = half * FACE_DIM;
            const uint32_t start = (from_col > col0) ? (from_col - col0) : 0;
            if (start >= FACE_DIM) {
                continue;
            }
            const uint32_t datum = (face_pair + half) * FACE_DIM * FACE_DIM + in_face_row * FACE_DIM + start;
            fill_pad_region(tile_base + datum * out_elem, (FACE_DIM - start) * out_elem, pad_word);
        }
    }
}

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
    constexpr uint32_t resident = get_compile_time_arg_val(9);         // A3/C14 zero-copy (1 = on)
    // R7: the output-format pad rewrite (see `fill_tile_pad`) and the geometry
    // it shares with the reader's pad body. `out_pad_fix == 0` erases all of it.
    constexpr uint32_t out_pad_fix = get_compile_time_arg_val(10);
    constexpr uint32_t out_pad_word = get_compile_time_arg_val(11);   // fill, OUTPUT format, replicated
    constexpr uint32_t pad_hp = get_compile_time_arg_val(12);         // PADDED rows per image
    constexpr uint32_t pad_h_real = get_compile_time_arg_val(13);     // REAL rows per image
    constexpr uint32_t pad_nimg = get_compile_time_arg_val(14);       // REAL images
    constexpr uint32_t pad_real_cols = get_compile_time_arg_val(15);  // REAL datum columns per row
    constexpr uint32_t tile_h = get_compile_time_arg_val(16);         // rows per tile
    constexpr uint32_t out_elem = get_compile_time_arg_val(17);       // OUTPUT element bytes
    // R9: the reader's two knobs, applied to the WRITE half — master.md B10
    // (this core's unicast VC) and A3 (the block-order decode). Both 0 by
    // default, which emits Refinement 8's writer byte-for-byte.
    constexpr uint32_t vc_mode = get_compile_time_arg_val(18);      // lever R9/B10 (1 = per-core VC)
    constexpr uint32_t block_order = get_compile_time_arg_val(19);  // lever R9/A3 (1 = row-major)
    constexpr auto dst_args = TensorAccessorArgs<20>();

    // B5 off-arm granularity: a tile is four faces, so the non-coalesced arm
    // issues one write per face instead of one per page.
    constexpr uint32_t FACES_PER_TILE = 4;
    constexpr uint32_t face_bytes = out_tile_bytes / FACES_PER_TILE;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1);
    const uint32_t nb = get_arg_val<uint32_t>(2);
    // R9/B10: this core's unicast VC — the SAME number the reader got, so a
    // core's two streams take matching lanes. Unlike the read side, the write
    // API honours it per call (`ncrisc_noc_fast_write` always programs
    // NOC_CMD_STATIC_VC), so it is passed straight to `noc_async_write`.
    const uint32_t write_vc = (vc_mode == 1) ? get_arg_val<uint32_t>(3) : NOC_UNICAST_WRITE_VC;

    // A3/C14 zero-copy: `cb_output_tiles` is ALIASED onto this core's own output
    // shard, so compute packed the tiles straight into their final L1 home —
    // there is nothing to write out. The writer exists only to drain the CB.
    // R7: the per-block pad geometry, shared by both writer paths. `r` is the
    // GLOBAL tile-row (b0 is the core's real block start on every path that can
    // be padded — a padded call never makes the INPUT shard resident, so it is
    // either streamed or a crossover, both of which carry a real b0).
    auto pad_rows_cols = [](uint32_t r, uint32_t c0, uint32_t t, uint32_t& real_rows, uint32_t& real_cols) {
        const uint32_t g0 = r * tile_h;
        const uint32_t img = g0 / pad_hp;
        const uint32_t row0 = g0 - img * pad_hp;
        real_rows = 0;
        if (img < pad_nimg && row0 < pad_h_real) {
            real_rows = pad_h_real - row0;
            if (real_rows > tile_h) {
                real_rows = tile_h;
            }
        }
        const uint32_t tile_col0 = (c0 + t) * 32;
        real_cols = 0;
        if (tile_col0 < pad_real_cols) {
            real_cols = pad_real_cols - tile_col0;
            if (real_cols > 32) {
                real_cols = 32;
            }
        }
    };

    if constexpr (resident == 1) {
        const uint32_t pages = nb * wt_block;
        cb_wait_front(cb_output_tiles, pages);
        if constexpr (out_pad_fix == 1) {
            // The tiles are already in their final L1 home (the aliased output
            // shard), so the rewrite happens in place before the CB is drained.
            const uint32_t l1_base = get_read_ptr(cb_output_tiles);
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t b = b0 + i;
                const uint32_t wchunk = b / nt_h;
                const uint32_t r = b - wchunk * nt_h;
                const uint32_t c0 = wchunk * wt_block;
                for (uint32_t t = 0; t < wt_block; ++t) {
                    uint32_t real_rows, real_cols;
                    pad_rows_cols(r, c0, t, real_rows, real_cols);
                    fill_tile_pad(
                        l1_base + (i * wt_block + t) * out_tile_bytes,
                        real_rows,
                        real_cols,
                        tile_h,
                        out_elem,
                        out_pad_word);
                }
            }
        }
        cb_pop_front(cb_output_tiles, pages);
        return;
    }

    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t i = 0; i < nb; ++i) {
        const uint32_t b = b0 + i;
        // R9/A3: the same decode the reader uses — the two kernels walk one
        // block space, so the order is a shared compile-time constant.
        const uint32_t wchunk = (block_order == 1) ? (b % n_wchunks) : (b / nt_h);
        const uint32_t r = (block_order == 1) ? (b / n_wchunks) : (b - wchunk * nt_h);
        const uint32_t w = (wchunk == n_wchunks - 1) ? wt_tail : wt_block;  // tiles this block
        const uint32_t c0 = wchunk * wt_block;                              // first tile-column

        cb_wait_front(cb_output_tiles, w);
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);
        const uint32_t first_page = r * Wt + c0;

        for (uint32_t t = 0; t < w; ++t) {
            if constexpr (out_pad_fix == 1) {
                uint32_t real_rows, real_cols;
                pad_rows_cols(r, c0, t, real_rows, real_cols);
                fill_tile_pad(l1_addr, real_rows, real_cols, tile_h, out_elem, out_pad_word);
            }
            if constexpr (stub_write == 0) {
                if constexpr (coalesce_writes == 1) {
                    noc_async_write(l1_addr, dst.get_noc_addr(first_page + t), out_tile_bytes, noc_index, write_vc);
                } else {
                    for (uint32_t f = 0; f < FACES_PER_TILE; ++f) {
                        noc_async_write(
                            l1_addr + f * face_bytes,
                            dst.get_noc_addr(first_page + t, f * face_bytes),
                            face_bytes,
                            noc_index,
                            write_vc);
                    }
                }
            }
            l1_addr += out_tile_bytes;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, w);
    }

    // R9/B10, the writer's half of the hand-back. A plain `noc_async_write`
    // reprograms NOC_CTRL on every call, so this is not load-bearing for the
    // next *write* — but a later kernel that issues `*_with_state` writes
    // without its own `set_state` would inherit our VC, and `noc_init` does not
    // run between launches (`brisc.cc` calls it only on a NoC-MODE change). The
    // reader arm proved that hazard is real (a byte-identical control arm
    // measured 1.14x after a VC arm), so both halves hand the register back.
    if constexpr (vc_mode == 1) {
        noc_async_write_one_packet_set_state(dst.get_noc_addr(0), out_tile_bytes, noc_index, NOC_UNICAST_WRITE_VC);
    }
}
