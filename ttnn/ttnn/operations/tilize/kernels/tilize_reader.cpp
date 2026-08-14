// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NCRISC / NOC0).
//
// Reads ROW_MAJOR sticks from an interleaved tensor into cb_input_sticks, one
// block at a time. A block is 1 tile-row x WT_CHUNK tile-columns (op_design.md
// §1): TILE_H sticks of WT_CHUNK*32*elem bytes each, written at L1 stride
// row_bytes, with ONE barrier per block (master.md B7).
//
// Two regimes, selected by a compile-time arg (op_design.md §5.1):
//
//   R_ALIGNED — the hot path. Delegates verbatim to the library helper
//               dataflow_kernel_lib::read_sticks_for_tilize<TILE granularity>.
//               Requires ONE PAGE PER STICK (the helper walks consecutive page
//               ids as consecutive rows), so the host selects the general R_PAD
//               loop instead when the source shard is narrower than a row
//               (src_row_pages > 1 — the cross-spec L1 gather). With an aligned
//               source that loop fills nothing: valid_bytes == row_bytes.
//
//   R_PAD     — HELPER SUBSTITUTION, justified: the library helper cannot fill.
//               Its contract (tilize_helpers_dataflow.hpp:50-52) states that for
//               a partial block "untouched rows contain stale data", and
//               .inl:120-123 reads only row_bytes while advancing L1 by the
//               padded stride, leaving the W tail untouched. There is no fill
//               parameter and no other kernel_lib helper covers a value-filled
//               read, while the pad oracle compares the pad region exactly. The
//               pad branch therefore uses raw dataflow (TensorAccessor +
//               noc_async_read + an L1 fill), keeping the helper's block
//               structure and one-barrier-per-block policy.
//
// Orthogonal to the fill regime, two PLACEMENT regimes (op_design.md §5.2):
//
//   P_ACCESSOR    — TensorAccessor over interleaved DRAM/L1 (or a non-local L1
//                   shard). Issues the reads described above.
//   P_LOCAL_SHARD — cb_input_sticks is ALIASED on this core's resident RM shard,
//                   so the block is already in L1 at exactly the layout tilize
//                   wants (tile_h sticks x WT_CHUNK*32 elements, WT_CHUNK being
//                   the whole shard width). The reader issues NO NoC read at
//                   all: it only publishes the pages. Re-reading them through a
//                   TensorAccessor would re-fetch data already resident in L1.
//
// ... and two work assignments (see the host's W_BLOCKS / W_REGION):
//
//   W_BLOCKS — a contiguous range of the global W-chunk-major block index.
//   W_REGION — the core's own shard tile region, walked tile-row-major with the
//              W chunk INNERMOST (that order is the shard's own linear tile
//              order, which is what lets an aliased output CB chunk its width).

//   R_RETILE  — the input is ALREADY TILE layout, at a DIFFERENT tile height
//               (Refinement 5). The reader walks FACES instead of sticks: it
//               stages whole source tile pages into an L1 scratch CB (a page is
//               always page-aligned, which is what keeps this addressable on
//               Blackhole, where DRAM alignment is 64 B) and then moves face rows
//               out of them into cb_input_sticks as ordinary row-major sticks.
//               From there the pipeline is unchanged: compute tilizes those
//               sticks into the REQUESTED tile height, which is what makes the
//               retile a reader-only change.
//
//               A tiled source cannot back the input CB directly (that CB holds
//               row-major sticks, by the tilize helper's contract), so a retile
//               always reads its source through the accessor — including a local
//               shard. That is not the "tolerated, not implemented" accessor read
//               the sharded refinements forbid: the bytes have to be permuted, so
//               there is nothing to consume in place.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
// fill_l1_with_val: the alignment-aware, sub-word-replicating L1 fill (shared with
// the writer, which stamps the same pad region in the OUTPUT element format after
// the cast). copy_l1_words: the retile path's local face-row move. One source for
// both L1 primitives.
#include "ttnn/ttnn/operations/tilize/kernels/tilize_fill.hpp"

namespace {

using tilize_kernels::copy_l1_words;
using tilize_kernels::fill_l1_with_val;

// Read `n_bytes` of ONE row-major source row, starting `byte_off` bytes into it,
// into L1 at `l1_addr` — the single source for source addressing on the accessor
// path (op_design.md §5.2, the cross-spec L1 gather).
//
// A ROW_MAJOR page is one row (stick) when the tensor is interleaved OR its shard
// spans the whole row: then `row_pages == 1`, the page id IS the row index and
// this is ONE transfer, byte-identical to Phase 0. A shard NARROWER than the row
// (WIDTH / BLOCK sharded, read from another core's L1) makes a page one SHARD
// row, so a row is `row_pages` pages of `page_bytes` and a span may cross page
// boundaries — issued as one transfer per page slice. Both cases keep the
// caller's one-barrier-per-block policy (master.md B7); nothing is barriered here.
template <uint32_t page_bytes, uint32_t row_pages, typename Accessor>
FORCE_INLINE void read_row_span(
    const Accessor& accessor, uint32_t row, uint32_t byte_off, uint32_t n_bytes, uint32_t l1_addr) {
    if constexpr (row_pages == 1) {
        noc_async_read(accessor.get_noc_addr(row, byte_off), l1_addr, n_bytes);
    } else {
        uint32_t page_in_row = byte_off / page_bytes;
        uint32_t page = row * row_pages + page_in_row;
        uint32_t off = byte_off - page_in_row * page_bytes;
        while (n_bytes > 0) {
            uint32_t n = page_bytes - off;
            if (n > n_bytes) {
                n = n_bytes;
            }
            noc_async_read(accessor.get_noc_addr(page, off), l1_addr, n);
            l1_addr += n;
            n_bytes -= n;
            ++page;
            off = 0;
        }
    }
}

// Issue ONE tile-row's worth of source rows (tile_h transfers of `row_bytes`)
// into consecutive L1 slots. Nothing is barriered here — the caller owns the
// barrier policy (master.md B7), which is what lets the trid loop below defer it.
//
// `one_packet` (master.md B6): a transfer that fits NOC_MAX_BURST_SIZE can skip
// the any-length loop and take the cheap single-packet issue path.
// `vc` (master.md B10): the read-REQUEST virtual channel. Readers that share a
// route serialize first-come-first-serve on one VC; spreading requests over the
// unicast VCs is the documented way to break that.
template <uint32_t row_bytes, uint32_t tile_h, bool one_packet, typename Accessor>
FORCE_INLINE void issue_tile_row(
    const Accessor& accessor, uint32_t first_row, uint32_t byte_off, uint32_t l1_addr, uint32_t vc) {
    for (uint32_t r = 0; r < tile_h; ++r) {
        const uint64_t src = accessor.get_noc_addr(first_row + r, byte_off);
        if constexpr (one_packet) {
            noc_async_read_one_packet(src, l1_addr, row_bytes, noc_index, vc);
        } else {
            noc_async_read(src, l1_addr, row_bytes, noc_index, vc);
        }
        l1_addr += row_bytes;  // aligned path: the L1 stride IS row_bytes
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_sticks = 0;
    // R_RETILE only: L1 scratch holding the staged SOURCE tile pages. Reader-owned
    // (never pushed or popped — it has no consumer), sized to one block by the host.
    constexpr uint32_t cb_retile_stage = 1;
    constexpr uint32_t TILE_W = 32;  // a tile is always 32 wide (hardware fact, not a knob)

    constexpr uint32_t regime = get_compile_time_arg_val(0);
    constexpr uint32_t placement = get_compile_time_arg_val(1);  // P_ACCESSOR / P_LOCAL_SHARD
    constexpr uint32_t work_mode = get_compile_time_arg_val(2);  // W_BLOCKS / W_REGION
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(4);  // the W block factor
    constexpr uint32_t nt_h = get_compile_time_arg_val(5);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(6);  // W chunks per shard row (W_REGION)
    constexpr uint32_t nth_per_img = get_compile_time_arg_val(7);
    constexpr uint32_t h_in = get_compile_time_arg_val(8);
    constexpr uint32_t n_img_in = get_compile_time_arg_val(9);
    constexpr uint32_t w_in_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(11);
    // Classification ablation (op_design.md §9.1): drop the NoC payload, keep
    // every CB reserve/push and the loop trip counts. Always 0 in production.
    constexpr uint32_t ablate_dm = get_compile_time_arg_val(12);
    // Source page geometry (op_design.md §5.2). src_row_pages == 1 means one page
    // IS one stick (interleaved, or a shard as wide as the row) — the Phase-0
    // identity `page id == row index`. > 1 is the cross-spec gather: the source
    // shard is NARROWER than a row, so a row is src_row_pages pages.
    constexpr uint32_t src_page_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t src_row_pages = get_compile_time_arg_val(14);
    // Refinement-3 levers on the interleaved aligned path (see the custom-loop
    // note below): B6 one-packet issue, B8 trid double-issue, B10 per-reader VC.
    constexpr uint32_t read_one_packet = get_compile_time_arg_val(15);
    constexpr uint32_t read_trid = get_compile_time_arg_val(16);
    constexpr uint32_t read_vc_enable = get_compile_time_arg_val(17);
    // Refinement 4: 1 = the WRITER re-stamps every pad position in the output
    // element format (the widening-cast path), which makes this reader's own
    // input-format fill dead work — the two fill regions are the same set, derived
    // from the same h_in / w_in / image geometry. Skipping it leaves stale bytes in
    // the input CB's pad positions, which the tilize permutes into output positions
    // the writer overwrites before the tile leaves L1. Host gate: `out_fill`.
    constexpr uint32_t skip_pad_fill = get_compile_time_arg_val(18);
    // --- Refinement 5: the RETILE path (regime R_RETILE) ---------------------
    // in_tile_h is the SOURCE tile's height (the output's is `tile_h` above);
    // `wt` is the tile-column count, which the retile reader needs because its
    // source pages are TILES, indexed `tile_row * wt + tile_col` — the plain
    // reader's pages are sticks and never need it. Both are 0/unused elsewhere.
    constexpr uint32_t in_tile_h = get_compile_time_arg_val(19);
    constexpr uint32_t wt = get_compile_time_arg_val(20);
    constexpr auto src_args = TensorAccessorArgs<21>();

    // Every byte quantity below derives from the WT_CHUNK knob — one source.
    constexpr uint32_t row_bytes = wt_chunk * TILE_W * elem_bytes;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t pad_word = get_arg_val<uint32_t>(3);
    const uint32_t tile_row0 = get_arg_val<uint32_t>(4);     // W_REGION: region origin
    const uint32_t col_off_base = get_arg_val<uint32_t>(5);  // W_REGION: byte offset in a stick
    const uint32_t read_vc = get_arg_val<uint32_t>(6);       // B10: this core's read-request VC
    const uint32_t tile_col0 = get_arg_val<uint32_t>(7);     // W_REGION origin, in TILE columns (R_RETILE)

    if (num_blocks == 0) {
        return;
    }

    if constexpr (placement == 1 /* P_LOCAL_SHARD */) {
        // ── ZERO-COPY ────────────────────────────────────────────────────
        // The CB *is* the resident shard: the data is already in L1 in exactly
        // the layout tilize consumes. Publish the pages, issue no NoC traffic.
        for (uint32_t i = 0; i < num_blocks; ++i) {
            cb_reserve_back(cb_input_sticks, wt_chunk);
            cb_push_back(cb_input_sticks, wt_chunk);
        }
        return;
    }

    const auto accessor = TensorAccessor(src_args, src_addr);

    if constexpr (ablate_dm) {
        // Payload removed, synchronization intact: same block count, same CB
        // handshake, same barrier — no reads. Compute runs on whatever is in L1.
        for (uint32_t i = 0; i < num_blocks; ++i) {
            cb_reserve_back(cb_input_sticks, wt_chunk);
            volatile uint32_t touch = get_write_ptr(cb_input_sticks);
            (void)touch;
            noc_async_read_barrier();
            cb_push_back(cb_input_sticks, wt_chunk);
        }
        return;
    }

    if constexpr (regime == 0) {
        // ── R_ALIGNED ────────────────────────────────────────────────────
        if constexpr (work_mode == 1 /* W_REGION */) {
            // The core's own shard region. With one W chunk per row (the common
            // case — a shard whose width fits the CB budget) the whole region is
            // ONE helper call over num_blocks*TILE_H contiguous sticks, so the
            // batched read of the interleaved path is preserved.
            if constexpr (n_chunks == 1) {
                dataflow_kernel_lib::
                    read_sticks_for_tilize<cb_input_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                        accessor,
                        /*total_num_rows=*/num_blocks * tile_h,
                        /*row_bytes=*/row_bytes,
                        /*start_page=*/tile_row0 * tile_h,
                        /*byte_offset_within_page=*/col_off_base);
            } else {
                for (uint32_t i = 0; i < num_blocks; ++i) {
                    const uint32_t r = i / n_chunks;
                    const uint32_t c = i - r * n_chunks;
                    dataflow_kernel_lib::
                        read_sticks_for_tilize<cb_input_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                            accessor,
                            /*total_num_rows=*/tile_h,
                            /*row_bytes=*/row_bytes,
                            /*start_page=*/(tile_row0 + r) * tile_h,
                            /*byte_offset_within_page=*/col_off_base + c * row_bytes);
                }
            }
        } else if constexpr (read_one_packet || read_trid || read_vc_enable) {
            // ── custom aligned loop (master.md B6 / B8 / B10) ─────────────
            // HELPER SUBSTITUTION, justified and MEASURED. read_sticks_for_tilize
            // issues a plain noc_async_read per row and one plain
            // noc_async_read_barrier per block; its contract exposes NO
            // transaction id (B8), NO request VC (B10) and NO one-packet
            // selector (B6), so none of the three levers this refinement prices
            // can be expressed through it. The helper stays the DEFAULT arm —
            // with all three levers off this branch is compiled out entirely and
            // the kernel takes the helper call below verbatim, so the OFF arm is
            // literally the Phase-0 code and the substitution itself is what the
            // bench measures.
            constexpr uint32_t in_tile_bytes = get_tile_size(cb_input_sticks);
            constexpr uint32_t slot_bytes = wt_chunk * in_tile_bytes;
            constexpr bool one_packet = read_one_packet && (row_bytes <= NOC_MAX_BURST_SIZE);
            const uint32_t vc = read_vc_enable ? read_vc : NOC_UNICAST_WRITE_VC;

            if constexpr (read_trid) {
                // B8 double-issue: block i's reads are issued BEFORE block i-1's
                // barrier, so a request is always in flight across the block
                // boundary instead of the NoC draining at every barrier.
                //
                // The host only sets this lever when the input CB is EXACTLY two
                // blocks deep (CB_DEPTH == 2, NT_BLK == 1), so the write pointer
                // alternates between two fixed slots and no wrap arithmetic is
                // needed — cb_reserve_back still provides all the flow control.
                const uint32_t slot_base = get_write_ptr(cb_input_sticks);
                uint32_t slot = 0;
                uint32_t trid_issue = 1, trid_wait = 1;
                bool in_flight = false;
                for (uint32_t i = 0; i < num_blocks; ++i) {
                    const uint32_t block = start_block + i;
                    const uint32_t wc = block / nt_h;
                    const uint32_t row = block % nt_h;

                    // Room for the still-unpushed in-flight block AND this one.
                    cb_reserve_back(cb_input_sticks, in_flight ? 2 * wt_chunk : wt_chunk);
                    noc_async_read_set_trid(trid_issue);
                    issue_tile_row<row_bytes, tile_h, one_packet>(
                        accessor, row * tile_h, wc * row_bytes, slot_base + slot * slot_bytes, vc);
                    slot ^= 1;
                    trid_issue ^= 3;  // alternate 1 <-> 2

                    if (in_flight) {
                        noc_async_read_barrier_with_trid(trid_wait);
                        cb_push_back(cb_input_sticks, wt_chunk);
                        trid_wait ^= 3;
                    }
                    in_flight = true;
                }
                noc_async_read_barrier_with_trid(trid_wait);  // drain the last block
                cb_push_back(cb_input_sticks, wt_chunk);
                // Leave the command buffer's packet tag at 0 for the next
                // kernel (the writer's twin MUST do this — brisck.cc:91 asserts
                // it on the write cmd bufs; the read cmd buf is not in that
                // check, but the hygiene is the same).
                noc_async_read_set_trid(0);
            } else {
                for (uint32_t i = 0; i < num_blocks; ++i) {
                    const uint32_t block = start_block + i;
                    const uint32_t wc = block / nt_h;
                    const uint32_t row = block % nt_h;

                    cb_reserve_back(cb_input_sticks, wt_chunk);
                    issue_tile_row<row_bytes, tile_h, one_packet>(
                        accessor, row * tile_h, wc * row_bytes, get_write_ptr(cb_input_sticks), vc);
                    noc_async_read_barrier();
                    cb_push_back(cb_input_sticks, wt_chunk);
                }
            }
        } else {
            // Blocks are W-chunk-major, so a run of consecutive blocks that shares
            // one W chunk is one helper call over run*TILE_H contiguous sticks.
            uint32_t block = start_block;
            uint32_t remaining = num_blocks;
            while (remaining > 0) {
                const uint32_t wc = block / nt_h;
                const uint32_t row = block % nt_h;
                uint32_t run = nt_h - row;
                if (run > remaining) {
                    run = remaining;
                }
                dataflow_kernel_lib::
                    read_sticks_for_tilize<cb_input_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                        accessor,
                        /*total_num_rows=*/run * tile_h,
                        /*row_bytes=*/row_bytes,
                        /*start_page=*/row * tile_h,
                        /*byte_offset_within_page=*/wc * row_bytes);
                block += run;
                remaining -= run;
            }
        }
    } else if constexpr (regime == 2 /* R_RETILE */) {
        // ── R_RETILE ─────────────────────────────────────────────────────
        // Stage whole SOURCE tile pages, then move face rows into the CB as
        // ordinary row-major sticks. Everything below the stage is a local L1
        // permutation, so no NoC alignment constraint reaches the face geometry.
        constexpr uint32_t FACE_W = 16;
        // A tile's face height is 16 on a full tile and the tile height itself on
        // a tiny one (tile.cpp TILE_FACE_HW_CHOICES) — the same rule the writer's
        // pad stamp uses.
        // `if constexpr` in a NON-template function still requires the discarded
        // branch to be well formed, and `in_tile_h` is 0 off the retile path — so
        // every constant here is written against a guarded height rather than the
        // raw arg (a bare `/ in_tile_h` is a compile error on every other cell).
        constexpr uint32_t src_tile_h = in_tile_h ? in_tile_h : 1;
        constexpr uint32_t src_face_h = src_tile_h < FACE_W ? src_tile_h : FACE_W;
        constexpr uint32_t faces_per_row = TILE_W / FACE_W;  // 2 (a tile is 32 wide)
        constexpr uint32_t src_tile_bytes = src_tile_h * TILE_W * elem_bytes;
        constexpr uint32_t face_row_bytes = FACE_W * elem_bytes;
        // Source tile-rows one OUTPUT tile-row spans. Both heights are powers of
        // two <= 32, so one divides the other and this is exact.
        constexpr uint32_t src_rows_per_block = (tile_h + src_tile_h - 1) / src_tile_h;

        const uint32_t stage_base = get_write_ptr(cb_retile_stage);
        // Consecutive blocks of a core share a W chunk and march up the tile-rows,
        // so when the OUTPUT tile is shorter than the source one (tile_h <
        // in_tile_h) several blocks read the same staged pages. Caching the last
        // staged (tile-row, tile-column) is what stops a 32 -> 1 retile fetching
        // each source page 32 times.
        uint32_t staged_row = 0xFFFFFFFFu, staged_col = 0xFFFFFFFFu;

        for (uint32_t i = 0; i < num_blocks; ++i) {
            uint32_t row, col;  // OUTPUT tile-row, first source tile-column
            if constexpr (work_mode == 1 /* W_REGION */) {
                const uint32_t r = i / n_chunks;
                row = tile_row0 + r;
                col = tile_col0 + (i - r * n_chunks) * wt_chunk;
            } else {
                const uint32_t block = start_block + i;
                row = block % nt_h;
                col = (block / nt_h) * wt_chunk;
            }
            const uint32_t src_row0 = (row * tile_h) / src_tile_h;

            if (src_row0 != staged_row || col != staged_col) {
                uint32_t addr = stage_base;
                for (uint32_t t = 0; t < src_rows_per_block; ++t) {
                    for (uint32_t k = 0; k < wt_chunk; ++k) {
                        // A whole tile PAGE, so the transfer is page-aligned on
                        // every arch (master.md B5 as a side effect).
                        noc_async_read(accessor.get_noc_addr((src_row0 + t) * wt + col + k), addr, src_tile_bytes);
                        addr += src_tile_bytes;
                    }
                }
                noc_async_read_barrier();  // one barrier per staged block (B7)
                staged_row = src_row0;
                staged_col = col;
            }

            cb_reserve_back(cb_input_sticks, wt_chunk);
            uint32_t l1_addr = get_write_ptr(cb_input_sticks);
            for (uint32_t r = 0; r < tile_h; ++r) {
                const uint32_t src_row = (row * tile_h + r) / src_tile_h - src_row0;  // staged tile index
                const uint32_t rr = (row * tile_h + r) % src_tile_h;                  // row inside it
                const uint32_t face_row = rr / src_face_h;
                const uint32_t row_in_face = rr % src_face_h;
                for (uint32_t k = 0; k < wt_chunk; ++k) {
                    const uint32_t tile_addr = stage_base + (src_row * wt_chunk + k) * src_tile_bytes;
                    for (uint32_t fc = 0; fc < faces_per_row; ++fc) {
                        copy_l1_words<face_row_bytes>(
                            tile_addr + ((face_row * faces_per_row + fc) * src_face_h + row_in_face) * face_row_bytes,
                            l1_addr + (k * TILE_W + fc * FACE_W) * elem_bytes);
                    }
                }
                l1_addr += row_bytes;
            }
            cb_push_back(cb_input_sticks, wt_chunk);
        }
    } else {
        // ── R_PAD ────────────────────────────────────────────────────────
        for (uint32_t i = 0; i < num_blocks; ++i) {
            // block -> (tile-row, byte offset), per work assignment
            uint32_t row, col_off;
            if constexpr (work_mode == 1 /* W_REGION */) {
                const uint32_t r = i / n_chunks;
                row = tile_row0 + r;
                col_off = col_off_base + (i - r * n_chunks) * row_bytes;
            } else {
                const uint32_t block = start_block + i;
                row = block % nt_h;
                col_off = (block / nt_h) * row_bytes;
            }
            const uint32_t img = row / nth_per_img;
            const uint32_t row_in_img = (row % nth_per_img) * tile_h;

            // Bytes of real data this block's rows carry (W tail beyond it).
            uint32_t valid_bytes = 0;
            if (col_off < w_in_bytes) {
                valid_bytes = w_in_bytes - col_off;
                if (valid_bytes > row_bytes) {
                    valid_bytes = row_bytes;
                }
            }

            cb_reserve_back(cb_input_sticks, wt_chunk);
            uint32_t l1_addr = get_write_ptr(cb_input_sticks);

            for (uint32_t r = 0; r < tile_h; ++r) {
                const uint32_t src_row = row_in_img + r;
                // H tail and whole pad tiles: no source row at all.
                const uint32_t n_read = (img < n_img_in && src_row < h_in) ? valid_bytes : 0;
                if (n_read > 0) {
                    read_row_span<src_page_bytes, src_row_pages>(
                        accessor, img * h_in + src_row, col_off, n_read, l1_addr);
                }
                if constexpr (!skip_pad_fill) {
                    if (n_read < row_bytes) {
                        fill_l1_with_val<elem_bytes>(l1_addr + n_read, row_bytes - n_read, pad_word);
                    }
                }
                l1_addr += row_bytes;
            }

            noc_async_read_barrier();
            cb_push_back(cb_input_sticks, wt_chunk);
        }
    }
}
