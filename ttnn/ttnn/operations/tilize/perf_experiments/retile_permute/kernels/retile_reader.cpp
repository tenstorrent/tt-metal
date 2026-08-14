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
// PERMANENT per-stage instrumentation (never remove — free when the profiler is
// off). Zones are split reserve / issue / barrier so a NoC stage that is
// RISC-bound on transaction count is distinguishable from one that is genuinely
// waiting on the fabric (.claude/references/device-zone-scope-attribution.md §4).
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
// fill_l1_with_val: the alignment-aware, sub-word-replicating L1 fill (shared with
// the writer, which stamps the same pad region in the OUTPUT element format after
// the cast). copy_l1_words: the retile path's local face-row move. One source for
// both L1 primitives.
#include "ttnn/ttnn/operations/tilize/kernels/tilize_fill.hpp"
// StatefulRead (master.md B13) and BlockIndex (master.md D21) — shared with the
// writer so each lever has exactly one implementation. Both are OFF-by-default
// arms that compile to the prior code.
#include "ttnn/ttnn/operations/tilize/kernels/tilize_noc.hpp"

// ── BAKE-OFF SELECTOR ────────────────────────────────────────────────────────
// Set by the generated per-arm shim (see bench_retile_permute.py). 0 == the op's
// current approach, reconstructed verbatim.
#ifndef RETILE_VARIANT
#define RETILE_VARIANT 0
#endif

namespace {

// L1 -> L1 copy, `nonvolatile` selecting whether the compiler is allowed to
// pipeline the loads ahead of the stores.
//
// RAW LLK / raw-dataflow note: the op's own primitive (tilize_fill.hpp
// `copy_l1_words`) uses `volatile tt_l1_ptr` pointers, which forces every load
// to complete before the next store issues — an rv32 L1 access is ~10 cycles, so
// a volatile load/store pair cannot overlap. Nothing in ttnn/cpp/ttnn/kernel_lib
// offers an L1->L1 block move at all (the dataflow helpers all address a
// TENSOR through an accessor), so this bench measures the primitive directly.
template <uint32_t n_bytes, bool nonvolatile>
FORCE_INLINE void copy_bytes(uint32_t src_addr, uint32_t dst_addr) {
    static_assert(n_bytes % 4 == 0, "L1 word copy needs a 4-byte multiple");
    if constexpr (nonvolatile) {
        // The staged bytes were landed by the NoC, which the compiler cannot
        // see; the explicit clobber keeps these loads below the read barrier.
        asm volatile("" ::: "memory");
        auto* src = reinterpret_cast<tt_l1_ptr uint32_t*>(src_addr);
        auto* dst = reinterpret_cast<tt_l1_ptr uint32_t*>(dst_addr);
#pragma GCC unroll 8
        for (uint32_t i = 0; i < n_bytes / 4; ++i) {
            dst[i] = src[i];
        }
        asm volatile("" ::: "memory");
    } else {
        tilize_kernels::copy_l1_words<n_bytes>(src_addr, dst_addr);
    }
}

using tilize_kernels::BlockIndex;
using tilize_kernels::copy_l1_words;
using tilize_kernels::fill_l1_with_val;
using tilize_kernels::StatefulRead;

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
//
// `issue` is the B13 stateful-read state (master.md B13): it carries the last
// programmed NoC endpoint across calls, so a run of transfers that share a
// source core costs one command-buffer register write less each. That run
// exists on the cross-core L1 gather (a source shard lives on ONE core, and a
// block's TILE_H rows all come from it) and never on an interleaved source.
template <uint32_t page_bytes, uint32_t row_pages, bool stateful, uint32_t packet_bytes, typename Accessor>
FORCE_INLINE void read_row_span(
    const Accessor& accessor,
    StatefulRead<stateful, packet_bytes>& issue,
    uint32_t row,
    uint32_t byte_off,
    uint32_t n_bytes,
    uint32_t l1_addr) {
    if constexpr (row_pages == 1) {
        issue.read(accessor.get_noc_addr(row, byte_off), l1_addr, n_bytes);
    } else {
        uint32_t page_in_row = byte_off / page_bytes;
        uint32_t page = row * row_pages + page_in_row;
        uint32_t off = byte_off - page_in_row * page_bytes;
        while (n_bytes > 0) {
            uint32_t n = page_bytes - off;
            if (n > n_bytes) {
                n = n_bytes;
            }
            issue.read(accessor.get_noc_addr(page, off), l1_addr, n);
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
// `stateful` (master.md B13): route the issue through the set_state/with_state
// pair instead. The stateful API carries no VC parameter, so the host never
// turns B13 and B10 on together (B10 ships parked anyway).
template <uint32_t row_bytes, uint32_t tile_h, bool one_packet, bool stateful, uint32_t packet_bytes, typename Accessor>
FORCE_INLINE void issue_tile_row(
    const Accessor& accessor,
    StatefulRead<stateful, packet_bytes>& issue,
    uint32_t first_row,
    uint32_t byte_off,
    uint32_t l1_addr,
    uint32_t vc) {
    for (uint32_t r = 0; r < tile_h; ++r) {
        const uint64_t src = accessor.get_noc_addr(first_row + r, byte_off);
        if constexpr (stateful) {
            issue.read(src, l1_addr, row_bytes);
        } else if constexpr (one_packet) {
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
    // --- Refinement 6 levers (master.md B13 / D21) ---------------------------
    // `read_state`: issue reads through set_state/with_state, caching the NoC
    // endpoint (see tilize_noc.hpp). `precomp_index`: take this core's
    // (tile-row, W chunk) origin from the host and step it, instead of paying a
    // div/mod per block. Both 0 => this kernel is byte-identical to Refinement 5.
    constexpr uint32_t read_state = get_compile_time_arg_val(21);
    constexpr uint32_t precomp_index = get_compile_time_arg_val(22);
    constexpr auto src_args = TensorAccessorArgs<23>();

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
    // D21: the host's decomposition of `start_block` (W_BLOCKS only; 0 off the
    // lever, where BlockIndex recomputes it per block instead).
    const uint32_t block_row0 = get_arg_val<uint32_t>(8);
    const uint32_t block_wc0 = get_arg_val<uint32_t>(9);

    if (num_blocks == 0) {
        return;
    }

    if constexpr (placement == 1 /* P_LOCAL_SHARD */) {
        // ── ZERO-COPY ────────────────────────────────────────────────────
        // The CB *is* the resident shard: the data is already in L1 in exactly
        // the layout tilize consumes. Publish the pages, issue no NoC traffic.
        for (uint32_t i = 0; i < num_blocks; ++i) {
            {
                // Pure back-pressure: the data is already resident, so anything
                // this reserve costs is the WRITER not draining (§5 of the
                // attribution doc), never a read.
                MaybeDeviceZoneScope("reader_reserve");
                cb_reserve_back(cb_input_sticks, wt_chunk);
            }
            cb_push_back(cb_input_sticks, wt_chunk);
        }
        return;
    }

    const auto accessor = TensorAccessor(src_args, src_addr);
    // B13 state (one per kernel: every read below shares the read command
    // buffer, so they share the endpoint the state programs).
    StatefulRead<read_state != 0> issue;

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
                // OCCUPANCY, not payload: read_sticks_for_tilize owns the CB
                // handshake AND the barrier internally, so this number folds
                // reserve + issue + barrier together. It cannot be split
                // without editing the helper — the cumulative ablation
                // (ABLATE["dm_read"]) is what separates payload from wait here.
                MaybeDeviceZoneScope("reader_helper");
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
                    MaybeDeviceZoneScope("reader_helper");
                    dataflow_kernel_lib::
                        read_sticks_for_tilize<cb_input_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                            accessor,
                            /*total_num_rows=*/tile_h,
                            /*row_bytes=*/row_bytes,
                            /*start_page=*/(tile_row0 + r) * tile_h,
                            /*byte_offset_within_page=*/col_off_base + c * row_bytes);
                }
            }
        } else if constexpr (read_one_packet || read_trid || read_vc_enable || read_state) {
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
            // B13 on the aligned path: every transfer here is exactly row_bytes,
            // so when it fits one packet the LENGTH goes into the state as well
            // and the lever composes with B6 rather than replacing it.
            StatefulRead<read_state != 0, (one_packet ? row_bytes : 0)> issue_rows;

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
                BlockIndex<precomp_index != 0, nt_h> idx;
                idx.init(start_block, block_row0, block_wc0);
                for (uint32_t i = 0; i < num_blocks; ++i) {
                    idx.seek(i);
                    const uint32_t wc = idx.wc;
                    const uint32_t row = idx.row;
                    idx.advance();

                    {
                        // Room for the still-unpushed in-flight block AND this one.
                        MaybeDeviceZoneScope("reader_reserve");
                        cb_reserve_back(cb_input_sticks, in_flight ? 2 * wt_chunk : wt_chunk);
                    }
                    {
                        MaybeDeviceZoneScope("reader_issue");
                        noc_async_read_set_trid(trid_issue);
                        issue_tile_row<row_bytes, tile_h, one_packet, read_state != 0>(
                            accessor, issue_rows, row * tile_h, wc * row_bytes, slot_base + slot * slot_bytes, vc);
                    }
                    slot ^= 1;
                    trid_issue ^= 3;  // alternate 1 <-> 2

                    if (in_flight) {
                        {
                            MaybeDeviceZoneScope("reader_barrier");
                            noc_async_read_barrier_with_trid(trid_wait);
                        }
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
                BlockIndex<precomp_index != 0, nt_h> idx;
                idx.init(start_block, block_row0, block_wc0);
                for (uint32_t i = 0; i < num_blocks; ++i) {
                    idx.seek(i);
                    const uint32_t wc = idx.wc;
                    const uint32_t row = idx.row;
                    idx.advance();

                    {
                        MaybeDeviceZoneScope("reader_reserve");
                        cb_reserve_back(cb_input_sticks, wt_chunk);
                    }
                    {
                        // RISC-serial issue cost: address generation + command
                        // buffer writes, scaling with TRANSACTION COUNT.
                        MaybeDeviceZoneScope("reader_issue");
                        issue_tile_row<row_bytes, tile_h, one_packet, read_state != 0>(
                            accessor, issue_rows, row * tile_h, wc * row_bytes, get_write_ptr(cb_input_sticks), vc);
                    }
                    {
                        // Time the fabric still owed us once issue finished.
                        MaybeDeviceZoneScope("reader_barrier");
                        noc_async_read_barrier();
                    }
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
                MaybeDeviceZoneScope("reader_helper");  // occupancy — see the n_chunks==1 note above
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
        // ══ ISOLATED BAKE-OFF: the RETILE permutation ════════════════════
        // Everything above this point is a VERBATIM copy of the op's reader so
        // the arms differ only in the permutation. `RETILE_VARIANT` selects:
        //
        //   0 baseline        the op's current approach (staged pages ->
        //                     row-major sticks, one volatile 32 B word copy per
        //                     (row, tile, face)), reconstructed verbatim.
        //   1 rm_coalesce     same row-major intermediate, but emitting the
        //                     LONGEST run contiguous in BOTH source and
        //                     destination. See the contiguity note below: that
        //                     run is 16 elements unless src_face_h == 1.
        //   2 rm_stage_direct no staging at all — when in_tile_h == 1 a source
        //                     tile PAGE *is* a row-major run, so the DRAM read
        //                     lands at its final address and there is no copy.
        //   3 direct_tile     staged pages -> the OUTPUT TILE directly (compute
        //                     becomes a no-op). The output face is a contiguous
        //                     run of a source face, so the run is rows_per_run
        //                     face-rows instead of one.
        //   4 direct_dram     DRAM -> the OUTPUT TILE. Same runs as (3), but the
        //                     NoC lands them: no staging, no CPU copy at all.
        //   5 noc_loopback    baseline geometry, but each L1->L1 face-row move
        //                     is a local noc_async_read instead of rv32 stores.
        //   6 nonvolatile     baseline geometry, non-volatile word copy so the
        //                     compiler may pipeline loads ahead of stores.
        //   7 direct_tile_nv  (3) + the non-volatile copy.
        //
        // Stage whole SOURCE tile pages, then move face rows into the CB as
        // ordinary row-major sticks. Everything below the stage is a local L1
        // permutation, so no NoC alignment constraint reaches the face geometry.
        constexpr uint32_t cb_output_tiles = 16;  // variants 3/4/7 produce this CB directly
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

        // ── run geometry ────────────────────────────────────────────────
        // (a) ROW-MAJOR intermediate. Stepping the destination by one face row
        //     (16 elements) steps the SOURCE by src_face_h face rows, because a
        //     tile's two faces are stored one whole face apart. So source and
        //     destination advance together only when src_face_h == 1, i.e. only
        //     for a 1-row source tile. Every other retile is stride-blocked at
        //     16 elements in this intermediate — the baseline's copy length is
        //     already maximal there, and coalescing has to change the SCHEME
        //     (variants 3/4) rather than the copy.
        constexpr bool rm_contiguous = (src_face_h == 1);
        // With src_face_h == 1 the whole block collapses: staged tile (t,k) sits
        // at (t*wt_chunk+k)*src_tile_bytes and row-major (r,k) at
        // r*row_bytes + k*TILE_W*elem — the same map, so it is ONE copy.
        constexpr uint32_t rm_run_bytes = rm_contiguous ? tile_h * row_bytes : face_row_bytes;

        // (b) OUTPUT TILE destination. Output face (ofr,ofc) holds out rows
        //     [ofr*out_face_h, +out_face_h) of column half ofc, row-major, 16
        //     elements per row — and those rows come from ONE source face as
        //     long as they stay inside it. Both heights are powers of two and
        //     the face origin is aligned, so the run is exactly
        //     min(out_face_h, src_face_h) face rows.
        constexpr uint32_t out_face_h = tile_h < FACE_W ? tile_h : FACE_W;
        constexpr uint32_t out_face_rows = tile_h / out_face_h;
        constexpr uint32_t out_tile_bytes_v = tile_h * TILE_W * elem_bytes;
        constexpr uint32_t rows_per_run = out_face_h < src_face_h ? out_face_h : src_face_h;
        constexpr uint32_t runs_per_face = out_face_h / rows_per_run;
        constexpr uint32_t tile_run_bytes = rows_per_run * face_row_bytes;

        const uint32_t stage_base = get_write_ptr(cb_retile_stage);
        // Consecutive blocks of a core share a W chunk and march up the tile-rows,
        // so when the OUTPUT tile is shorter than the source one (tile_h <
        // in_tile_h) several blocks read the same staged pages. Caching the last
        // staged (tile-row, tile-column) is what stops a 32 -> 1 retile fetching
        // each source page 32 times.
        uint32_t staged_row = 0xFFFFFFFFu, staged_col = 0xFFFFFFFFu;

        BlockIndex<precomp_index != 0, nt_h> idx;
        idx.init(start_block, block_row0, block_wc0);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            uint32_t row, col;  // OUTPUT tile-row, first source tile-column
            if constexpr (work_mode == 1 /* W_REGION */) {
                const uint32_t r = i / n_chunks;
                row = tile_row0 + r;
                col = tile_col0 + (i - r * n_chunks) * wt_chunk;
            } else {
                idx.seek(i);
                row = idx.row;
                col = idx.wc * wt_chunk;
                idx.advance();
            }
            const uint32_t src_row0 = (row * tile_h) / src_tile_h;
            const uint32_t out_row0 = row * tile_h;  // this block's first OUTPUT row

            // ── V2: no staging — the DRAM page lands at its final RM address ──
            if constexpr (RETILE_VARIANT == 2 && rm_contiguous) {
                {
                    MaybeDeviceZoneScope("reader_reserve");
                    cb_reserve_back(cb_input_sticks, wt_chunk);
                }
                const uint32_t l1_base = get_write_ptr(cb_input_sticks);
                {
                    MaybeDeviceZoneScope("retile_stage_issue");
                    for (uint32_t t = 0; t < src_rows_per_block; ++t) {
                        for (uint32_t k = 0; k < wt_chunk; ++k) {
                            issue.read(
                                accessor.get_noc_addr((src_row0 + t) * wt + col + k),
                                l1_base + (t * wt_chunk + k) * src_tile_bytes,
                                src_tile_bytes);
                        }
                    }
                }
                {
                    MaybeDeviceZoneScope("retile_stage_barrier");
                    noc_async_read_barrier();
                }
                cb_push_back(cb_input_sticks, wt_chunk);
                continue;
            }

            // ── V4: DRAM -> OUTPUT TILE, the NoC does the permutation ────────
            if constexpr (RETILE_VARIANT == 4) {
                {
                    MaybeDeviceZoneScope("reader_reserve");
                    cb_reserve_back(cb_output_tiles, wt_chunk);
                }
                const uint32_t out_base = get_write_ptr(cb_output_tiles);
                {
                    MaybeDeviceZoneScope("retile_stage_issue");
                    for (uint32_t k = 0; k < wt_chunk; ++k) {
                        const uint32_t dst_tile = out_base + k * out_tile_bytes_v;
                        for (uint32_t ofr = 0; ofr < out_face_rows; ++ofr) {
                            for (uint32_t ofc = 0; ofc < faces_per_row; ++ofc) {
                                const uint32_t dst_face =
                                    dst_tile + (ofr * faces_per_row + ofc) * out_face_h * face_row_bytes;
                                for (uint32_t rn = 0; rn < runs_per_face; ++rn) {
                                    const uint32_t out_row = out_row0 + ofr * out_face_h + rn * rows_per_run;
                                    const uint32_t si = out_row % src_tile_h;
                                    const uint32_t page = (out_row / src_tile_h) * wt + col + k;
                                    const uint32_t off =
                                        ((si / src_face_h * faces_per_row + ofc) * src_face_h + si % src_face_h) *
                                        face_row_bytes;
                                    issue.read(
                                        accessor.get_noc_addr(page, off),
                                        dst_face + rn * tile_run_bytes,
                                        tile_run_bytes);
                                }
                            }
                        }
                    }
                }
                {
                    MaybeDeviceZoneScope("retile_stage_barrier");
                    noc_async_read_barrier();
                }
                cb_push_back(cb_output_tiles, wt_chunk);
                continue;
            }

            if (src_row0 != staged_row || col != staged_col) {
                {
                    MaybeDeviceZoneScope("retile_stage_issue");
                    uint32_t addr = stage_base;
                    for (uint32_t t = 0; t < src_rows_per_block; ++t) {
                        for (uint32_t k = 0; k < wt_chunk; ++k) {
                            // A whole tile PAGE, so the transfer is page-aligned on
                            // every arch (master.md B5 as a side effect).
                            issue.read(accessor.get_noc_addr((src_row0 + t) * wt + col + k), addr, src_tile_bytes);
                            addr += src_tile_bytes;
                        }
                    }
                }
                {
                    MaybeDeviceZoneScope("retile_stage_barrier");
                    noc_async_read_barrier();  // one barrier per staged block (B7)
                }
                staged_row = src_row0;
                staged_col = col;
            }

            if constexpr (RETILE_VARIANT == 3 || RETILE_VARIANT == 7 || RETILE_VARIANT == 8) {
                // ── V3/V7: staged pages -> the OUTPUT TILE ──────────────────
                // The row-major round trip (and with it the compute tilize) is
                // gone: an output face is a contiguous run of a source face, so
                // this emits rows_per_run face-rows per copy instead of one.
                {
                    MaybeDeviceZoneScope("reader_reserve");
                    cb_reserve_back(cb_output_tiles, wt_chunk);
                }
                {
                    MaybeDeviceZoneScope("retile_permute");
                    const uint32_t out_base = get_write_ptr(cb_output_tiles);
                    for (uint32_t k = 0; k < wt_chunk; ++k) {
                        const uint32_t dst_tile = out_base + k * out_tile_bytes_v;
                        for (uint32_t ofr = 0; ofr < out_face_rows; ++ofr) {
                            for (uint32_t ofc = 0; ofc < faces_per_row; ++ofc) {
                                const uint32_t dst_face =
                                    dst_tile + (ofr * faces_per_row + ofc) * out_face_h * face_row_bytes;
                                for (uint32_t rn = 0; rn < runs_per_face; ++rn) {
                                    const uint32_t out_row = out_row0 + ofr * out_face_h + rn * rows_per_run;
                                    const uint32_t si = out_row % src_tile_h;
                                    const uint32_t st = out_row / src_tile_h - src_row0;
                                    const uint32_t src =
                                        stage_base + (st * wt_chunk + k) * src_tile_bytes +
                                        ((si / src_face_h * faces_per_row + ofc) * src_face_h + si % src_face_h) *
                                            face_row_bytes;
                                    if constexpr (RETILE_VARIANT == 8) {
                                        // V8: staged page -> output tile, but the
                                        // local NoC lands the run. Costs ONE DRAM
                                        // page read per source tile instead of one
                                        // DRAM transaction per run, which is what
                                        // V4 pays when the run is tiny.
                                        noc_async_read(
                                            get_noc_addr(src), dst_face + rn * tile_run_bytes, tile_run_bytes);
                                    } else {
                                        copy_bytes<tile_run_bytes, RETILE_VARIANT == 7>(
                                            src, dst_face + rn * tile_run_bytes);
                                    }
                                }
                            }
                        }
                    }
                    if constexpr (RETILE_VARIANT == 8) {
                        noc_async_read_barrier();
                    }
                }
                cb_push_back(cb_output_tiles, wt_chunk);
                continue;
            }

            {
                MaybeDeviceZoneScope("reader_reserve");
                cb_reserve_back(cb_input_sticks, wt_chunk);
            }
            {
                // The retile's real payload: a pure L1->L1 face-row permutation,
                // no NoC at all. Refinement 5 classified this path L1-store-bound;
                // this zone is what makes that claim re-checkable per block.
                MaybeDeviceZoneScope("retile_permute");
                uint32_t l1_addr = get_write_ptr(cb_input_sticks);
                if constexpr (RETILE_VARIANT == 1 && rm_contiguous) {
                    // V1 on the one geometry where the row-major intermediate IS
                    // contiguous: the whole block is a single run.
                    copy_bytes<rm_run_bytes, false>(stage_base, l1_addr);
                } else {
                    for (uint32_t r = 0; r < tile_h; ++r) {
                        const uint32_t src_row = (row * tile_h + r) / src_tile_h - src_row0;  // staged tile index
                        const uint32_t rr = (row * tile_h + r) % src_tile_h;                  // row inside it
                        const uint32_t face_row = rr / src_face_h;
                        const uint32_t row_in_face = rr % src_face_h;
                        for (uint32_t k = 0; k < wt_chunk; ++k) {
                            const uint32_t tile_addr = stage_base + (src_row * wt_chunk + k) * src_tile_bytes;
                            for (uint32_t fc = 0; fc < faces_per_row; ++fc) {
                                const uint32_t src =
                                    tile_addr +
                                    ((face_row * faces_per_row + fc) * src_face_h + row_in_face) * face_row_bytes;
                                const uint32_t dst = l1_addr + (k * TILE_W + fc * FACE_W) * elem_bytes;
                                if constexpr (RETILE_VARIANT == 5) {
                                    // V5: let the NoC do the local L1->L1 move.
                                    noc_async_read(get_noc_addr(src), dst, face_row_bytes);
                                } else {
                                    copy_bytes<face_row_bytes, RETILE_VARIANT == 6>(src, dst);
                                }
                            }
                        }
                        l1_addr += row_bytes;
                    }
                    if constexpr (RETILE_VARIANT == 5) {
                        noc_async_read_barrier();
                    }
                }
            }
            cb_push_back(cb_input_sticks, wt_chunk);
        }
    } else {
        // ── R_PAD ────────────────────────────────────────────────────────
        BlockIndex<precomp_index != 0, nt_h> idx;
        idx.init(start_block, block_row0, block_wc0);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            // block -> (tile-row, byte offset), per work assignment
            uint32_t row, col_off;
            if constexpr (work_mode == 1 /* W_REGION */) {
                const uint32_t r = i / n_chunks;
                row = tile_row0 + r;
                col_off = col_off_base + (i - r * n_chunks) * row_bytes;
            } else {
                idx.seek(i);
                row = idx.row;
                col_off = idx.wc * row_bytes;
                idx.advance();
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

            {
                MaybeDeviceZoneScope("reader_reserve");
                cb_reserve_back(cb_input_sticks, wt_chunk);
            }
            uint32_t l1_addr = get_write_ptr(cb_input_sticks);

            {
                // Issue + L1 fill are interleaved per source row by construction
                // (the fill is what makes the block whole), so they share one
                // zone; ABLATE["dm_read"] / `skip_pad_fill` separate them.
                MaybeDeviceZoneScope("reader_issue");
                for (uint32_t r = 0; r < tile_h; ++r) {
                    const uint32_t src_row = row_in_img + r;
                    // H tail and whole pad tiles: no source row at all.
                    const uint32_t n_read = (img < n_img_in && src_row < h_in) ? valid_bytes : 0;
                    if (n_read > 0) {
                        read_row_span<src_page_bytes, src_row_pages, read_state != 0>(
                            accessor, issue, img * h_in + src_row, col_off, n_read, l1_addr);
                    }
                    if constexpr (!skip_pad_fill) {
                        if (n_read < row_bytes) {
                            fill_l1_with_val<elem_bytes>(l1_addr + n_read, row_bytes - n_read, pad_word);
                        }
                    }
                    l1_addr += row_bytes;
                }
            }

            {
                MaybeDeviceZoneScope("reader_barrier");
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_sticks, wt_chunk);
        }
    }
}
