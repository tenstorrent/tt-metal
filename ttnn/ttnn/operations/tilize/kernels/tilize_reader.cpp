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
// the cast).
#include "ttnn/ttnn/operations/tilize/kernels/tilize_fill.hpp"
// StatefulRead (master.md B13) and BlockIndex (master.md D21) — shared with the
// writer so each lever has exactly one implementation. Both are OFF-by-default
// arms that compile to the prior code.
#include "ttnn/ttnn/operations/tilize/kernels/tilize_noc.hpp"

namespace {

using tilize_kernels::BlockIndex;
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
// `coal` (Perf 2): source sticks per NoC transfer. The aligned path's L1 stride
// IS row_bytes, so when the host proves `coal` consecutive sticks are one
// contiguous source range (an L1-sharded source whose page is a whole tensor
// row), they are also contiguous in the destination and the whole run is ONE
// transfer. coal == 1 is the per-stick issue, unchanged.
template <
    uint32_t row_bytes,
    uint32_t tile_h,
    uint32_t coal,
    bool one_packet,
    bool stateful,
    uint32_t packet_bytes,
    typename Accessor>
FORCE_INLINE void issue_tile_row(
    const Accessor& accessor,
    StatefulRead<stateful, packet_bytes>& issue,
    uint32_t first_row,
    uint32_t byte_off,
    uint32_t l1_addr,
    uint32_t vc) {
    constexpr uint32_t step = coal > tile_h ? tile_h : coal;  // never leave the block
    constexpr uint32_t xfer_bytes = step * row_bytes;
    for (uint32_t r = 0; r < tile_h; r += step) {
        const uint64_t src = accessor.get_noc_addr(first_row + r, byte_off);
        if constexpr (stateful) {
            issue.read(src, l1_addr, xfer_bytes);
        } else if constexpr (one_packet) {
            noc_async_read_one_packet(src, l1_addr, xfer_bytes, noc_index, vc);
        } else {
            noc_async_read(src, l1_addr, xfer_bytes, noc_index, vc);
        }
        l1_addr += xfer_bytes;  // aligned path: the L1 stride IS row_bytes
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_sticks_a = 0;
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
    // Perf 2: `read_ahead` is the issue-ahead window in blocks (0 or 1) and
    // `read_coalesce` the source sticks merged into one transfer (1, or tile_h).
    // Together they replaced B8's read-side `read_trid`, which was exactly this
    // loop with ZERO CB slack — measured baseline-or-worse on every cell.
    constexpr uint32_t read_ahead = get_compile_time_arg_val(16);
    constexpr uint32_t read_coalesce = get_compile_time_arg_val(17);
    constexpr uint32_t read_vc_enable = get_compile_time_arg_val(18);
    // Refinement 4: 1 = the WRITER re-stamps every pad position in the output
    // element format (the widening-cast path), which makes this reader's own
    // input-format fill dead work — the two fill regions are the same set, derived
    // from the same h_in / w_in / image geometry. Skipping it leaves stale bytes in
    // the input CB's pad positions, which the tilize permutes into output positions
    // the writer overwrites before the tile leaves L1. Host gate: `out_fill`.
    constexpr uint32_t skip_pad_fill = get_compile_time_arg_val(19);
    // --- Refinement 5: the RETILE path (regime R_RETILE) ---------------------
    // in_tile_h is the SOURCE tile's height (the output's is `tile_h` above);
    // `wt` is the tile-column count, which the retile reader needs because its
    // source pages are TILES, indexed `tile_row * wt + tile_col` — the plain
    // reader's pages are sticks and never need it. Both are 0/unused elsewhere.
    constexpr uint32_t in_tile_h = get_compile_time_arg_val(20);
    constexpr uint32_t wt = get_compile_time_arg_val(21);
    // --- Refinement 6 levers (master.md B13 / D21) ---------------------------
    // `read_state`: issue reads through set_state/with_state, caching the NoC
    // endpoint (see tilize_noc.hpp). `precomp_index`: take this core's
    // (tile-row, W chunk) origin from the host and step it, instead of paying a
    // div/mod per block. Both 0 => this kernel is byte-identical to Refinement 5.
    constexpr uint32_t read_state = get_compile_time_arg_val(22);
    constexpr uint32_t precomp_index = get_compile_time_arg_val(23);
    // --- Perf 2 (R_RETILE only): the retile-DIRECT reader --------------------
    // `retile_direct`: land the face permutation straight in the OUTPUT TILE
    // instead of routing it through a row-major intermediate. 0 only where the
    // host's one carve-out fires (a 1-row output tile — see the host).
    // `retile_direct_dram`: source each run from DRAM rather than from a staged
    // page (needs the run DRAM-alignable and >= the transaction floor).
    // `retile_cast`: a cast was requested, so the finished tile is left in the
    // INPUT dtype in cb_input_sticks for compute to convert.
    constexpr uint32_t retile_direct = get_compile_time_arg_val(24);
    constexpr uint32_t retile_direct_dram = get_compile_time_arg_val(25);
    constexpr uint32_t retile_cast = get_compile_time_arg_val(26);
    // Perf 2: block slots in the input CB (one group deeper than the output CB,
    // for the issue-ahead window). get_write_ptr only advances on push_back, so
    // the reader walks the ring itself while a read is outstanding.
    constexpr uint32_t in_cb_slots = get_compile_time_arg_val(27);
    // Perf 2 (cross-core L1 gather): the block's width IS one source shard row,
    // so a whole block is ONE contiguous transfer instead of tile_h per-row ones.
    constexpr uint32_t gather_coalesce = get_compile_time_arg_val(28);
    // Perf 2 SPLIT READER: `split_mode` 0 none / 1 dedicated dual-NoC / 2 shared
    // NOC_0 with per-RISC transaction ids; `phase` is which half this RISC owns.
    // Both DM kernels are THIS source file with only `phase` differing, so the two
    // halves cannot drift apart.
    constexpr uint32_t split_mode = get_compile_time_arg_val(29);
    constexpr uint32_t phase = get_compile_time_arg_val(30);
    // Each half owns every other block and publishes its OWN CB — one CB with two
    // issuers is not expressible (cb_push_back moves a single shared write
    // pointer, so ordering two producers needs a per-block semaphore handshake
    // that would re-serialize the very issue the split parallelizes).
    constexpr uint32_t cb_input_sticks_b = 3;
    constexpr uint32_t block_stride = split_mode ? 2 : 1;
    // On the shared-NOC_0 flavor each reader tags its own reads and barriers on
    // that id alone (per-id hardware state, so it is RISC-agnostic). 0 = the
    // ordinary any-transaction barrier.
    constexpr uint32_t split_trid = (split_mode == 2) ? (phase + 1) : 0;
    constexpr auto src_args = TensorAccessorArgs<31>();
    // This RISC's input CB: phase 0 (NCRISC) publishes the first, phase 1 (BRISC,
    // split path only) the second.
    constexpr uint32_t cb_input_sticks = phase ? cb_input_sticks_b : cb_input_sticks_a;

    constexpr uint32_t cb_output_tiles = 16;
    // Which CB this kernel publishes. On the direct path with no cast the reader
    // IS the producer of the finished output tiles and compute is empty; every
    // other path publishes row-major sticks (or an output-shaped tile awaiting a
    // cast) into cb_input_sticks, exactly as before.
    constexpr uint32_t reader_out_cb =
        (regime == 2 /* R_RETILE */ && retile_direct && !retile_cast) ? cb_output_tiles : cb_input_sticks;

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
        // The published CB follows the production path (`reader_out_cb`), or the
        // ablation would deadlock the pipeline it is supposed to leave intact.
        for (uint32_t i = phase; i < num_blocks; i += block_stride) {
            cb_reserve_back(reader_out_cb, wt_chunk);
            volatile uint32_t touch = get_write_ptr(reader_out_cb);
            (void)touch;
            noc_async_read_barrier();
            cb_push_back(reader_out_cb, wt_chunk);
        }
        return;
    }

    if constexpr (regime == 0) {
        // ── R_ALIGNED — ONE loop (Perf 2) ────────────────────────────────
        // This single loop replaces the five R_ALIGNED paths the op used to
        // carry: the W_REGION batched helper call, the W_REGION per-block helper
        // call, the W_BLOCKS run-batched helper calls, B8's two-slot trid
        // double-issue, and the plain custom loop. At `read_ahead == 0` and
        // `read_coalesce == 1` it issues exactly what the helper issued (same
        // accessor, same page order, same L1 stride, same one-group-per-push
        // granularity) and MEASURED FLAT against it on all eight regimes swept —
        // that control is what makes the unification honest rather than a
        // strawman baseline.
        //
        // HELPER SUBSTITUTION — dataflow_kernel_lib::read_sticks_for_tilize.
        // CLASS: capability. The helper owns its CB handshake AND its barrier
        // internally (one plain noc_async_read per stick, one plain
        // noc_async_read_barrier per block) and its contract exposes NO
        // transaction id, NO in-flight window and NO multi-stick transfer, so
        // neither thing this loop does is reachable through it at any argument.
        //
        // Two schedules, both host-derived, both measured:
        //   * `read_ahead == 1` keeps ONE group of reads outstanding across the
        //     block boundary over rotating trids, against ONE group of CB SLACK
        //     (IN_CB_EXTRA_DEPTH). 1.19x crossover, 1.24x tall, 1.18x interleaved
        //     W_BLOCKS, 1.18x uint8; flat at the DRAM floor, at one block per
        //     core, and on the smallest 2-tile cell. The slack is the mechanism:
        //     a window with cb_depth == ahead + 1 measured baseline-or-worse
        //     (that is precisely what B8's read half was, which is why it is
        //     gone), and ahead >= 2 regresses.
        //   * `read_coalesce == tile_h` merges a whole block's sticks into ONE
        //     transfer where the host proved them contiguous (an L1-sharded
        //     source whose page is a whole tensor row): 1.12-1.18x, on exactly
        //     the topology where issue-ahead loses.
        constexpr uint32_t in_tile_bytes = get_tile_size(cb_input_sticks);
        constexpr uint32_t slot_bytes = wt_chunk * in_tile_bytes;
        constexpr uint32_t coal = read_coalesce > tile_h ? tile_h : (read_coalesce ? read_coalesce : 1);
        constexpr bool one_packet = read_one_packet && (coal * row_bytes <= NOC_MAX_BURST_SIZE);
        constexpr uint32_t n_trid = read_ahead + 1;
        // The split's per-RISC transaction id, or the issue-ahead window's
        // rotating pair — never both (the host turns issue-ahead off on the split
        // path, which is the configuration the 1.50-1.65x was measured in).
        constexpr bool tag_trid = (read_ahead > 0) || (split_trid != 0);
        const uint32_t vc = read_vc_enable ? read_vc : NOC_UNICAST_WRITE_VC;
        // B13: every transfer on this path is the same length, so when it fits
        // one packet the LENGTH goes into the state too and B13 composes with B6.
        StatefulRead<read_state != 0, (one_packet ? coal * row_bytes : 0)> issue_rows;

        // The reader is this CB's only producer and starts on an empty CB, so the
        // write pointer at entry IS the ring base and the slot walk is exact.
        // Flow control still comes entirely from cb_reserve_back.
        const uint32_t slot_base = get_write_ptr(cb_input_sticks);
        uint32_t slot = 0;
        uint32_t trid_issue = 1, trid_wait = 1;
        uint32_t pending = 0;

        BlockIndex<precomp_index != 0, nt_h> idx;
        idx.init(start_block, block_row0, block_wc0);
        for (uint32_t i = phase; i < num_blocks; i += block_stride) {
            // block -> (first source stick, byte offset in the stick), per work
            // assignment. UNCHANGED from the loops this replaces.
            uint32_t row, col_off;
            if constexpr (work_mode == 1 /* W_REGION */) {
                const uint32_t r = (n_chunks == 1) ? i : i / n_chunks;
                row = tile_row0 + r;
                col_off = col_off_base + ((n_chunks == 1) ? 0 : (i - r * n_chunks) * row_bytes);
            } else {
                idx.seek(i);
                row = idx.row;
                col_off = idx.wc * row_bytes;
                idx.advance();
            }

            {
                // Room for every still-unpushed group AND the one about to be
                // issued — the slack group is what makes that reservable.
                MaybeDeviceZoneScope("reader_reserve");
                cb_reserve_back(cb_input_sticks, (pending + 1) * wt_chunk);
            }
            {
                // RISC-serial issue cost: address generation + command buffer
                // writes, scaling with TRANSACTION COUNT. (Note it also absorbs
                // fabric back-pressure — noc_async_read blocks inside this loop
                // when the NoC is saturated, so a large number here is not by
                // itself proof of RISC-bound issue work.)
                MaybeDeviceZoneScope("reader_issue");
                if constexpr (tag_trid) {
                    noc_async_read_set_trid(split_trid ? split_trid : trid_issue);
                }
                issue_tile_row<row_bytes, tile_h, coal, one_packet, read_state != 0>(
                    accessor, issue_rows, row * tile_h, col_off, slot_base + slot * slot_bytes, vc);
            }
            slot = (slot + 1 == in_cb_slots) ? 0 : slot + 1;
            if constexpr (read_ahead > 0) {
                trid_issue = (trid_issue == n_trid) ? 1 : trid_issue + 1;
            }
            ++pending;

            if (pending > read_ahead) {
                {
                    // Time the fabric still owed us once issue finished.
                    MaybeDeviceZoneScope("reader_barrier");
                    if constexpr (split_trid != 0) {
                        // Barrier on THIS RISC's own reads only — the partner
                        // reader shares NOC_0 and its transfers must not be waited
                        // on here (nor it on ours).
                        noc_async_read_barrier_with_trid(split_trid);
                    } else if constexpr (read_ahead > 0) {
                        noc_async_read_barrier_with_trid(trid_wait);
                        trid_wait = (trid_wait == n_trid) ? 1 : trid_wait + 1;
                    } else {
                        noc_async_read_barrier();
                    }
                }
                cb_push_back(cb_input_sticks, wt_chunk);
                --pending;
            }
        }
        // Drain whatever the window still holds. A core with fewer blocks than
        // the window never fills it, and this is what degenerates that case to
        // the plain schedule instead of hanging.
        while (pending > 0) {
            {
                MaybeDeviceZoneScope("reader_barrier");
                if constexpr (split_trid != 0) {
                    noc_async_read_barrier_with_trid(split_trid);
                } else if constexpr (read_ahead > 0) {
                    noc_async_read_barrier_with_trid(trid_wait);
                    trid_wait = (trid_wait == n_trid) ? 1 : trid_wait + 1;
                } else {
                    noc_async_read_barrier();
                }
            }
            cb_push_back(cb_input_sticks, wt_chunk);
            --pending;
        }
        if constexpr (tag_trid) {
            // Leave the command buffer's packet tag at 0 for the next kernel
            // (the same hygiene the writer's trid twin observes).
            noc_async_read_set_trid(0);
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

        // ── Perf 2: the DIRECT face-run geometry ─────────────────────────
        // Output face (ofr, ofc) holds output rows [ofr*out_face_h, +out_face_h)
        // of column half ofc, row-major, 16 elements per row. Those rows come
        // from ONE source face as long as they stay inside it, and both face
        // heights are powers of two with aligned origins, so the contiguous run
        // is exactly min(out_face_h, src_face_h) face rows — 8 face rows = 256 B
        // on a 32->8 bf16 retile, versus the legacy form's ONE face row (32 B).
        constexpr uint32_t out_face_h = tile_h < FACE_W ? tile_h : FACE_W;
        constexpr uint32_t out_face_rows = tile_h / (out_face_h ? out_face_h : 1);
        constexpr uint32_t out_tile_bytes_v = tile_h * TILE_W * elem_bytes;
        constexpr uint32_t rows_per_run = out_face_h < src_face_h ? out_face_h : src_face_h;
        constexpr uint32_t runs_per_face = out_face_h / (rows_per_run ? rows_per_run : 1);
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
            const uint32_t out_row0 = row * tile_h;

            // The staging pass exists for every form that does NOT read its runs
            // straight out of DRAM; under `retile_direct_dram` it compiles out
            // entirely and the scratch CB is never touched.
            if constexpr (!retile_direct_dram) {
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
            }

            {
                MaybeDeviceZoneScope("reader_reserve");
                cb_reserve_back(reader_out_cb, wt_chunk);
            }
            if constexpr (retile_direct) {
                // ── PERF 2 — the DIRECT face-run permutation ──────────────
                // RAW-NoC/HELPER SUBSTITUTION, justified and MEASURED (see the
                // host's `retile_direct` for the mechanism). The destination is
                // the OUTPUT TILE's own face layout, so each transfer is a run of
                // `rows_per_run` face rows instead of a single 32 B face row, and
                // the row-major intermediate — with its second L1 crossing and its
                // whole tilize compute pass — disappears.
                //
                // Measured on [1,1,1024,1024] bf16 32->8 DRAM->DRAM: the Perf-1
                // loopback form 41,949 -> 23,982 ns (1.72x). Wins on every
                // geometry measured except a 1-row output tile, which the host
                // carves out (`retile_direct` is 0 there and the legacy form
                // below runs). No helper in ttnn/cpp/ttnn/kernel_lib/ can express
                // "gather N strided runs into an output TILE's face layout with
                // one barrier for the batch" — see the changelog's Helper
                // bypasses table. Do NOT rewrite this back onto a helper.
                MaybeDeviceZoneScope("retile_permute");
                const uint32_t out_base = get_write_ptr(reader_out_cb);
                for (uint32_t k = 0; k < wt_chunk; ++k) {
                    const uint32_t dst_tile = out_base + k * out_tile_bytes_v;
                    for (uint32_t ofr = 0; ofr < out_face_rows; ++ofr) {
                        for (uint32_t ofc = 0; ofc < faces_per_row; ++ofc) {
                            const uint32_t dst_face =
                                dst_tile + (ofr * faces_per_row + ofc) * out_face_h * face_row_bytes;
                            for (uint32_t rn = 0; rn < runs_per_face; ++rn) {
                                const uint32_t out_row = out_row0 + ofr * out_face_h + rn * rows_per_run;
                                const uint32_t si = out_row % src_tile_h;
                                const uint32_t off =
                                    ((si / src_face_h * faces_per_row + ofc) * src_face_h + si % src_face_h) *
                                    face_row_bytes;
                                const uint32_t dst = dst_face + rn * tile_run_bytes;
                                if constexpr (retile_direct_dram) {
                                    const uint32_t page = (out_row / src_tile_h) * wt + col + k;
                                    issue.read(accessor.get_noc_addr(page, off), dst, tile_run_bytes);
                                } else {
                                    const uint32_t st = out_row / src_tile_h - src_row0;
                                    noc_async_read(
                                        get_noc_addr(stage_base + (st * wt_chunk + k) * src_tile_bytes + off),
                                        dst,
                                        tile_run_bytes);
                                }
                            }
                        }
                    }
                }
                // One barrier per block, as everywhere else in this op (B7).
                noc_async_read_barrier();
            } else {
                // ── the CARVE-OUT: a 1-row output tile ────────────────────
                // Reached only when the host set `retile_direct` to 0, which
                // happens for exactly one reason: `tile_h == 1`. There
                // out_face_h == 1, so a direct run is a single face row with no
                // reuse to win and the direct form MEASURED 0.79-0.89x (bf16
                // 32->1: 79,959-89,195 ns vs 70,788 here). tile_h == 2 is already
                // a 1.33x win for the direct form, so this exception is exactly
                // one tile height wide, not "small tiles".
                //
                // The retile's payload in this form: the face-row permutation
                // into a ROW-MAJOR intermediate, which the tilize compute then
                // re-tiles.
                //
                // PERF 1 — HELPER/PRIMITIVE SUBSTITUTION, justified and MEASURED.
                // This used to be `copy_l1_words<face_row_bytes>`, an rv32
                // load/store loop, and it was the single hottest stage in the whole
                // op: 85,718 of the 100,201 ns wall on a 32->8 retile of
                // [1,1,1024,1024] bf16 (86%), with the reader as a whole 83% of the
                // wall by cumulative payload ablation. Issuing each face-row move as
                // a LOCAL NoC read instead (source and destination are both this
                // core's own L1, so `get_noc_addr` resolves to a loopback address)
                // hands the copy to the NoC, which keeps several moves in flight
                // instead of one rv32 store at a time. Measured 99,849 -> 41,902 ns
                // (2.38x) on the focus case, and it wins on EVERY retile geometry
                // measured: 32->16/8/4/2/1 and 1/2/4/8/16->32, bf16 / fp32 / uint8,
                // interleaved and local-shard destinations.
                //
                // Why this and not the 4.19x direct-to-output-tile form: that one
                // needs `out_dtype == in_dtype` (it hands raw bytes to the writer,
                // so the packer's cast has no owner) plus a DRAM-alignment
                // predicate, i.e. a four-way dispatch. This form needs NO predicate
                // at all — it keeps the row-major intermediate and the real tilize
                // compute, so it is correct for casting retiles too. The faster
                // direct form is measured and recorded in changelog.md `## Perf 1`.
                //
                // No `ttnn/cpp/ttnn/kernel_lib/` helper covers an L1->L1 block move
                // (which is why `copy_l1_words` is op-local in the first place).
                MaybeDeviceZoneScope("retile_permute");
                uint32_t l1_addr = get_write_ptr(cb_input_sticks);
                for (uint32_t r = 0; r < tile_h; ++r) {
                    const uint32_t src_row = (row * tile_h + r) / src_tile_h - src_row0;  // staged tile index
                    const uint32_t rr = (row * tile_h + r) % src_tile_h;                  // row inside it
                    const uint32_t face_row = rr / src_face_h;
                    const uint32_t row_in_face = rr % src_face_h;
                    for (uint32_t k = 0; k < wt_chunk; ++k) {
                        const uint32_t tile_addr = stage_base + (src_row * wt_chunk + k) * src_tile_bytes;
                        for (uint32_t fc = 0; fc < faces_per_row; ++fc) {
                            noc_async_read(
                                get_noc_addr(
                                    tile_addr +
                                    ((face_row * faces_per_row + fc) * src_face_h + row_in_face) * face_row_bytes),
                                l1_addr + (k * TILE_W + fc * FACE_W) * elem_bytes,
                                face_row_bytes);
                        }
                    }
                    l1_addr += row_bytes;
                }
                // One barrier per block, as everywhere else in this op (master.md
                // B7). The moves must land before the block is published.
                noc_async_read_barrier();
            }
            cb_push_back(reader_out_cb, wt_chunk);
        }
    } else {
        // ── R_PAD ────────────────────────────────────────────────────────
        BlockIndex<precomp_index != 0, nt_h> idx;
        idx.init(start_block, block_row0, block_wc0);
        if constexpr (split_trid != 0) {
            noc_async_read_set_trid(split_trid);
        }
        for (uint32_t i = phase; i < num_blocks; i += block_stride) {
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
                // ── PERF 2: the WHOLE-BLOCK cross-core gather ─────────────
                // When the block's width IS one source shard row (the host sets
                // `gather_coalesce`), the block's tile_h source rows are tile_h
                // CONSECUTIVE pages of ONE source shard — page p lives on shard
                // `p % src_row_pages` at local row `p / src_row_pages` — so they
                // are contiguous in that core's L1, and the CB slot is contiguous
                // too because the block width is exactly the page. The whole
                // block is then ONE transfer instead of tile_h * slices.
                //
                // Measured end-to-end: reshard [1,1,1024,256] W2->H8 19,488 ->
                // 15,365 ns (1.27x); the 128 B-page gated plan 20,828 -> 13,512
                // (1.54x); a 1-tile-page source 2.91x; padded 1.22x. Flat, never
                // slower, once the per-row transfer is already >= 512 B (the row
                // form saturates source L1 egress there).
                //
                // The guard is the PAD guard, not a new one: a block whose rows
                // are all real and full-width takes the single transfer, and a
                // ragged block keeps the per-row read + fill below verbatim.
                // Raggedness is per block, so at most ONE tile-row per core ever
                // falls back, and the fallback writes the same addresses.
                bool coalesced = false;
                if constexpr (gather_coalesce) {
                    if (img < n_img_in && row_in_img + tile_h <= h_in && valid_bytes == row_bytes) {
                        issue.read(
                            accessor.get_noc_addr(
                                (img * h_in + row_in_img) * src_row_pages + col_off / src_page_bytes, 0),
                            l1_addr,
                            tile_h * row_bytes);
                        coalesced = true;
                    }
                }
                for (uint32_t r = 0; r < tile_h && !coalesced; ++r) {
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
                if constexpr (split_trid != 0) {
                    noc_async_read_barrier_with_trid(split_trid);
                } else {
                    noc_async_read_barrier();
                }
            }
            cb_push_back(cb_input_sticks, wt_chunk);
        }
        if constexpr (split_trid != 0) {
            noc_async_read_set_trid(0);
        }
    }
}
