// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NCRISC / NoC0).
//
// Modes, selected by compile-time args:
//
//   alias_mode == 1  (Path B, and Refinement 3's `alias_in` — zero-copy READ)
//       cb_rm_input is aliased onto the resident L1 ROW_MAJOR shard, so the
//       bytes are already at the CB's address. One cb_reserve_back /
//       cb_push_back arms the whole shard; there is no NoC traffic at all.
//       Refinement 3 reaches this branch for a sharded RM input whose OUTPUT is
//       interleaved as well (`path == "alias_in"`): the work split gives this core
//       exactly its own shard, so the unpacker reads the shard in place and only
//       the writer talks to the NoC. That deletes 12 749 ns of 19 734 on
//       `g_sharded_to_dram` (65 % of the runtime — the one-sided DM ablation).
//
//   coalesce_rows == 1  (Refinement 3, levers B5/B6 — bigger sharded reads)
//       A ROW_MAJOR-*sharded* source stores one page per logical row and exactly
//       ONE page column per shard, and `core_to_host_pages` pages a shard
//       row-major — so the 32 rows of a chunk-block are 32 CONSECUTIVE pages
//       inside a single core's L1. When the chunk covers the whole source page the
//       L1 destination is contiguous too, so the whole block is ONE read of
//       `32 * page_bytes` instead of 32 reads of `page_bytes`. `g_sharded_to_dram`
//       plans 128 B pages, 4x under the one-packet threshold, and its read leg is
//       issue-rate bound (~50 ns per read), which is exactly what this removes.
//
//   blocks_row_major == 1  (Refinement 3, the chunked `alias_out` order)
//       With the OUTPUT CB aliased onto the shard, CB page k IS shard tile k, and
//       a shard's tiles are stored row-major. So when the shard is wider than one
//       chunk the reader must iterate tile-row-OUTER / column-chunk-INNER, which
//       is the opposite of the generic path's order. With one chunk per core the
//       two orders coincide and this stays 0 (which is what keeps lever C7, whose
//       hand-off is chunk-outer, available on the crossover).
//
//   alias_mode == 0, split_read == 0  (Path A / C, single reader)
//       Chunk-outer, tile-row-inner. For each column chunk we hand a whole
//       tile-row band to dataflow_kernel_lib::read_sticks_for_tilize in TILE
//       granularity, which owns cb_reserve_back / 32 strided reads / one
//       noc_async_read_barrier / cb_push_back per block. `stateful_read`
//       selects StickReadMode::Stateful (lever B13) inside the helper.
//
//   alias_mode == 0, split_read == 1  (Path A / C, split reader — lever C7)
//       The 32 stick reads of each block are shared with BRISC (the writer
//       kernel), which parks in cb_wait_front for the whole read window
//       otherwise. NCRISC keeps sole ownership of the CB — a circular buffer
//       must have exactly ONE producer, so BRISC never reserves or pushes
//       cb_rm_input; it is handed the reserved window through two counting
//       semaphores:
//           NCRISC: reserve -> sem_reserve = blk+1 -> read half -> barrier
//                   -> wait sem_done >= blk+1 -> push
//           BRISC : wait sem_reserve >= blk+1 -> read half -> barrier
//                   -> sem_done = blk+1
//       Both semaphores are monotonic per-launch counters and both live in this
//       core's own L1, so set/wait are plain local loads and stores (no NoC
//       round trip). Requires depth == 1 so the reserved window is always the
//       CB base address, which is what BRISC's untouched get_write_ptr returns
//       (see the host gate in tilize_program_descriptor.py).
//
//   alias_mode == 0, prefetch_blocks == 2  (Path A / C — lever B8)
//       Trid double-issue. Each chunk-block's 32 stick reads carry one of two NoC
//       transaction ids, and the barrier is `noc_async_read_barrier_with_trid` on
//       the PREVIOUS id — so block i+1's reads are already in flight while block
//       i drains, instead of the NoC request queue emptying once per block. This
//       is the read-side analogue of the write side's `noc_async_writes_flushed()`
//       (Phase-0 verification fix #1): the writer can let writes stay in flight
//       because it only needs them to have DEPARTED, whereas the reader needs the
//       bytes PRESENT before it pushes, which is what the second trid buys.
//
//       NB the host gate keys on the BUSIEST core's block count, so on an uneven
//       split (`_split_contiguous` gives `total % parts` cores one extra unit) some
//       cores run this path with `total_blocks == 1`. That is safe by construction —
//       the prologue reserve covers the single push, the barrier parity matches the
//       trid the prologue set, and the tag is restored — and it is covered by
//       `test_tilize_refinement2.py::test_b8_is_bit_exact_on_an_uneven_split`.
//
//       It needs a THIRD CB window. `cb_reserve_back` does not move the write
//       pointer, so `get_write_ptr` returns the *current* block's window until
//       `cb_push_back` — the reader cannot ask the CB for the next block's address
//       before publishing the current one. The next window is therefore computed
//       from the CB base (`cb_base + (block % depth) * chunk_bytes`, exactly what
//       the FIFO's own pointer does after `depth` pushes) and its freedom is
//       guaranteed by reserving TWO windows. At depth 2 that reserve would demand
//       a fully drained CB and serialize compute behind the reader, hence
//       depth == 3 (host gate).
//
//   vc_spread == 1  (lever B10)
//       Program a per-core static unicast VC for this core's reads. In
//       DM_DEDICATED_NOC — what ReaderConfigDescriptor selects —
//       `noc_async_read`'s `read_req_vc` argument is DEAD: `ncrisc_noc_fast_read`
//       only writes NOC_CTRL under DM_DYNAMIC_NOC
//       (noc_nonblocking_api.h:415-437). NOC_CTRL is instead programmed once by
//       `noc_init` (static VC 1) and is STICKY, so one
//       `noc_async_read_one_packet_set_state<use_vc=true>` retargets every
//       subsequent read on this core — and must be undone before the kernel ends,
//       or the next program on this core inherits the custom VC and loses DRAM
//       bandwidth (same hazard the dram-sharded matmul reader documents).
//
//   stagger == 1  (Refinement 2b — per-core transaction-order rotation)
//       An interleaved tensor puts page p in DRAM bank `p % NUM_DRAM_BANKS`, and
//       every core issues its 32 stick reads in the SAME page order (row 0 first).
//       With `nt_h == 1` all cores read the same 32 pages, so at issue step r all
//       64 cores hit ONE bank while the other 11 idle -- the requests are spread
//       over the banks in aggregate but CLUSTERED in time. This lever rotates each
//       core's issue order by `row_rot` (and the writer's by `col_rot`), so step 0
//       is spread over the banks instead of piling onto one. It is a pure index
//       permutation: same transactions, same count, same size, same L1
//       destinations, zero extra state. The rotation is expressed as TWO
//       `read_stick_rows_for_tilize` calls over the two row runs, so the helper
//       still owns the address generation.
//
//   fanin_mode != 0  (Refinement 2b — whole-page staged read + L1 redistribution)
//       Only fires on the wide-short fan-in regime (`nt_h == 1`, one chunk-block per
//       core), where all `ncores` cores read disjoint `chunk_row_bytes` slices of the
//       SAME 32 source pages. That is a 64-way partial-page fan-in: measured
//       156.9 GB/s where the identical 512 B transaction reaching *private* pages
//       gets 179.3. This path breaks the coupling between "which bytes a core reads"
//       and "which tiles a core owns":
//
//         phase 1  each core reads ONE contiguous `piece_bytes` slice of ONE source
//                  page into cb_stage -- 32x fewer, 32x bigger DRAM transactions for
//                  exactly the same bytes. Cores are grouped `group_size == 32` (one
//                  per source row); group g stages source piece g.
//         phase 2  all-to-all ready handshake inside the group (one posted atomic
//                  inc per group-mate, including self -- a local `+=` would race with
//                  the remote atomics).
//         phase 3  each core PULLS its own `chunk_row_bytes` slice out of each
//                  group-mate's cb_stage into its own cb_rm_input window, stick r at
//                  offset r * chunk_row_bytes -- byte-identical to what the strided
//                  DRAM reader would have produced. Pull (not push) keeps every core
//                  the sole writer of its own CB memory.
//
//       Group-mate `r` is logical core (grp_x[r % grp_w], grp_y[r / grp_w]); the host
//       passes the group's PHYSICAL coordinate axes (compact: grp_w + grp_h words
//       instead of 2 * group_size). cb_stage's L1 address is the same on every core
//       in the range, so the remote source address is this core's own
//       `get_write_ptr(cb_stage)`.
//
//       `fanin_mode == 2` is a MEASUREMENT PROBE (bench only): phase 1 straight into
//       cb_rm_input with no exchange, so `/perf-ceiling-dm` can price the read-side
//       ceiling on its own. Output is garbage by design.
//
//       When the source is ROW_MAJOR-*sharded* with more than one page per
//       logical row (`row_page_stride > 1`) neither helper path can be used:
//       their page index advances by exactly 1 per row, hard-coding "one page ==
//       one full logical row", and the signature exposes no row-stride
//       parameter. The raw fallback below mirrors the helper's block structure
//       exactly (reserve chunk_wt, 32 reads, one barrier, push chunk_wt) so
//       lever B7 (one barrier per block) still holds.

//   zones == 1  (Refinement 3b lever 1 — MEASUREMENT VARIANT, bench only)
//       An instrumented copy of the shipped per-block read loop with a
//       `DeviceZoneScopedN` around each of its four stages (reserve / issue /
//       barrier / push), so `/perf-measure` can say WHICH RISC is waiting WHEN
//       instead of only how long each stage's payload costs. No ablation variant
//       can answer that: `no_dm` keeps the address-gen sink and `no_compute`
//       keeps the CB dance, so the residual falls between their attributions.
//       The shipped path is left byte-for-byte alone — this is a separate branch,
//       selected by a compile-time arg the host only sets from `TILIZE_ZONES=1`.
//       Correctness is unaffected (same reads, same CB counts) but the zone
//       writes perturb the timing, so it is never on in a shipped plan.

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_input = 0;
    constexpr uint32_t tile_height = 32;  // rows per tile-row block

    constexpr uint32_t alias_mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t row_page_stride = get_compile_time_arg_val(3);
    constexpr uint32_t source_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(5);
    // Perf-ablation only (TILIZE_SKIP_DM=1): drop the noc_async_read *payload* and
    // keep every CB op, barrier, handshake and loop trip count, so /perf-measure
    // can attribute time to the read stage. Never set on a correctness run.
    constexpr uint32_t skip_dm = get_compile_time_arg_val(6);
    constexpr uint32_t stateful_read = get_compile_time_arg_val(7);  // lever B13
    constexpr uint32_t split_read = get_compile_time_arg_val(8);     // lever C7
    constexpr uint32_t sem_reserve_id = get_compile_time_arg_val(9);
    constexpr uint32_t sem_done_id = get_compile_time_arg_val(10);
    constexpr uint32_t prefetch_blocks = get_compile_time_arg_val(11);  // lever B8
    constexpr uint32_t vc_spread = get_compile_time_arg_val(12);        // lever B10 (bitmask)
    constexpr bool read_vc_spread = (vc_spread & 1u) != 0;              // bit 0 == spread the reads
    constexpr uint32_t cb_depth = get_compile_time_arg_val(13);
    constexpr uint32_t trid_a = get_compile_time_arg_val(14);
    constexpr uint32_t trid_b = get_compile_time_arg_val(15);
    constexpr uint32_t default_read_vc = get_compile_time_arg_val(16);
    // --- Refinement 2b: whole-page staged read + L1 redistribution ---------------
    constexpr uint32_t fanin_mode = get_compile_time_arg_val(17);  // 0 off, 1 full, 2 read-probe
    constexpr uint32_t piece_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t group_size = get_compile_time_arg_val(19);
    constexpr uint32_t grp_w = get_compile_time_arg_val(20);
    constexpr uint32_t sem_fanin_id = get_compile_time_arg_val(21);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(22);
    constexpr uint32_t stagger = get_compile_time_arg_val(23);  // Refinement 2b
    // --- Refinement 3: crossover paths --------------------------------------------
    constexpr uint32_t coalesce_rows = get_compile_time_arg_val(24);     // levers B5/B6
    constexpr uint32_t blocks_row_major = get_compile_time_arg_val(25);  // chunked alias_out
    constexpr uint32_t read_group = get_compile_time_arg_val(26);        // lever B7' (below)
    // MEASUREMENT PROBE (bench only, garbage output): drop 31 of every 32
    // `accessor.get_noc_addr` calls in the read loop to price the
    // address-generation term on its own.
    //
    // MEASURED, AND THE INSTRUMENT IS REFUTED: 46 851 ns vs 16 743 (2.80x SLOWER).
    // Reusing ONE address per block does not only remove the arithmetic, it also
    // sends all 32 reads of the block to a SINGLE DRAM bank instead of spreading
    // them over 12 -- so it prices bank serialization, not address generation. The
    // number is kept because it is a third independent confirmation that this path's
    // read leg is bound by DRAM BANK PARALLELISM (i.e. bandwidth), not by issue rate:
    // collapsing 12 banks to 1 costs 2.8x, while halving the ISSUERS (lever C7) or
    // doubling the reads in flight (lever B8) each cost ~10 %.
    constexpr uint32_t addr_probe = get_compile_time_arg_val(27);
    // Refinement 3b lever 1: per-RISC Tracy timeline (bench only, see the header).
    constexpr uint32_t zones = get_compile_time_arg_val(28);
    constexpr auto src_args = TensorAccessorArgs<29>();

    using dataflow_kernel_lib::StickReadMode;
    constexpr StickReadMode read_mode = stateful_read ? StickReadMode::Stateful : StickReadMode::Generic;
    static_assert(!split_read || row_page_stride == 1, "the split reader needs one source page per logical row");
    static_assert(prefetch_blocks == 1 || prefetch_blocks == 2, "B8 double-issues exactly two transaction ids");
    static_assert(prefetch_blocks == 1 || !split_read, "B8 and C7 both own the read window; they are exclusive");
    // cb_depth >= 3 is EXACT, not conservative: `cb_reserve_back(2 * chunk_wt)`
    // guarantees blocks 0..b-(depth-2) are popped, and block b+1's window last held
    // block b+1-depth, so depth 3 gives precisely the needed guarantee with zero
    // margin. Do not relax it (any depth > 3 is also sound).
    static_assert(prefetch_blocks == 1 || cb_depth >= 3, "B8 needs a third CB window (see the header)");
    // The host would otherwise size the CB to 3 windows and then silently get the
    // raw strided fallback (correct output, lever quietly lost, no diagnostic).
    static_assert(prefetch_blocks == 1 || row_page_stride == 1, "B8 needs one source page per logical row");
    // Refinement 2b owns the whole read path for its regime, so it is exclusive with
    // every other read-path lever (each of which reshapes the same 32 stick reads).
    static_assert(
        !fanin_mode || (row_page_stride == 1 && !split_read && prefetch_blocks == 1 && !stateful_read),
        "the fan-in redistribution replaces the stick reads; B13/C7/B8 cannot also own them");
    static_assert(fanin_mode != 1 || group_size % grp_w == 0, "the fan-in group must be a core rectangle");
    // The rotation owns the row loop, so it cannot coexist with a lever that also
    // owns it (B8 flattens the block sequence, C7 splits it, B13 forces bank-major).
    static_assert(
        !stagger || (prefetch_blocks == 1 && !split_read && !stateful_read && !fanin_mode),
        "the read-order rotation owns the row loop; B8/C7/B13/fan-in cannot also own it");
    // Same diagnostic gap B8 guards against: with a multi-page source row the branch
    // condition below is false, so the host would report the lever ON while the kernel
    // silently took the raw strided fallback -- correct output, lever quietly lost.
    static_assert(!stagger || row_page_stride == 1, "the read-order rotation needs one source page per row");
    // Refinement 3. The coalesced read replaces the whole 32-read row loop with one
    // transaction, so every lever that reshapes that loop is exclusive with it; and
    // it is only VALID when the chunk covers exactly one source page (otherwise the
    // 32 pages it folds together are not contiguous in the source shard's L1).
    static_assert(
        !coalesce_rows || (chunk_row_bytes == source_page_bytes && !stateful_read && !split_read &&
                           prefetch_blocks == 1 && !stagger && !fanin_mode),
        "the coalesced sharded read owns the row loop, and needs chunk == one source page");
    // The row-major block order and lever C7's chunk-outer hand-off cannot both own
    // the block sequence (the host keeps C7 to one chunk per core, where the two
    // orders coincide and this flag is 0).
    static_assert(!blocks_row_major || !split_read, "row-major block order and the C7 hand-off are exclusive");
    // Lever C7 is a two-party protocol and THIS kernel is the party that reserves the
    // window and signals `sem_reserve`. Every branch that returns before doing so
    // leaves the writer spinning forever, so each one needs a tripwire: the aliased
    // read (which has no reads at all -- `ttnn-static-analyzer` F1, a hang reachable
    // from a plain public call), and the two Refinement-3 read-loop disjuncts that do
    // not already have one (F2, latent).
    static_assert(!alias_mode || !split_read, "an aliased read has no half to hand to BRISC");
    static_assert(read_group == 1 || !split_read, "the grouped read loop does not signal the C7 hand-off");
    static_assert(!addr_probe || !split_read, "the address probe does not signal the C7 hand-off");
    static_assert(
        !blocks_row_major || (prefetch_blocks == 1 && !fanin_mode && !stagger),
        "row-major block order owns the block sequence; B8/fan-in/rotation cannot also own it");
    // The timeline variant re-implements the plain per-block loop, so it is only
    // meaningful (and only correct) where that loop is what ships.
    static_assert(
        !zones || (!alias_mode && !fanin_mode && !coalesce_rows && !split_read && prefetch_blocks == 1 && !stagger &&
                   !addr_probe && read_group == 1 && row_page_stride == 1 && vc_spread == 0),
        "the per-RISC timeline instruments the plain per-block read loop only");

    if constexpr (alias_mode) {
        // Data is already resident at the CB address — just hand it to compute.
        cb_reserve_back(cb_rm_input, shard_tiles);
        cb_push_back(cb_rm_input, shard_tiles);
        return;
    } else {
        const uint32_t src_addr = get_arg_val<uint32_t>(0);
        const uint32_t start_row = get_arg_val<uint32_t>(1);
        const uint32_t num_rows = get_arg_val<uint32_t>(2);
        const uint32_t chunk_start = get_arg_val<uint32_t>(3);
        const uint32_t chunk_count = get_arg_val<uint32_t>(4);

        const auto accessor = TensorAccessor(src_args, src_addr);

        // --- lever B10: retarget this core's reads onto its own static VC ------
        // NOC_CTRL is sticky and `noc_async_read` never rewrites it in dedicated
        // mode, so one armed set_state moves every read below onto `read_vc`.
        if constexpr (read_vc_spread) {
            const uint32_t read_vc = get_arg_val<uint32_t>(5);
            noc_async_read_one_packet_set_state<true>(accessor.get_noc_addr(start_row), chunk_row_bytes, read_vc);
        }

        if constexpr (zones) {
            // --- Refinement 3b lever 1: the per-RISC timeline ---------------------
            // Same four stages the shipped loop runs, each in its own zone:
            //   RD-RESV  blocked waiting for compute to free a CB window
            //   RD-ISSUE address generation + 32 noc_async_read commands
            //   RD-BARR  draining this block's reads (the DRAM service time that
            //            the issue stage did not already hide)
            //   RD-PUSH  publishing the window
            // Each zone needs its OWN scope: the macro declares a variable named
            // `zone`, so two in one scope do not compile.
            const uint32_t blocks = num_rows / tile_height;
            const uint32_t total = blocks * chunk_count;
            constexpr uint32_t window_bytes = chunk_wt * get_tile_size(cb_rm_input);
            const uint32_t cb_base = get_write_ptr(cb_rm_input);
            for (uint32_t idx = 0; idx < total; ++idx) {
                const uint32_t block = idx % blocks;
                const uint32_t c = idx / blocks;
                const uint32_t row0 = start_row + block * tile_height;
                const uint32_t byte_offset = (chunk_start + c) * chunk_row_bytes;
                {
                    DeviceZoneScopedN("RD-RESV");
                    cb_reserve_back(cb_rm_input, chunk_wt);
                }
                {
                    // `skip_dm` keeps the 32 address-generation calls and drops only
                    // the noc_async_read commands, so RD-ISSUE(no_dm) is the pure
                    // address-generation term and RD-ISSUE(full) - RD-ISSUE(no_dm) is
                    // the command programming plus the NoC back-pressure. That split
                    // is the whole point: `noc_async_read` spins on
                    // `noc_cmd_buf_ready`, so a DRAM-bound read shows up INSIDE the
                    // issue loop, not in the barrier.
                    DeviceZoneScopedN("RD-ISSUE");
                    if constexpr (skip_dm) {
                        for (uint32_t row = 0; row < tile_height; ++row) {
                            volatile uint32_t sink =
                                static_cast<uint32_t>(accessor.get_noc_addr(row0 + row, byte_offset));
                            (void)sink;
                        }
                    } else {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<read_mode, 1>(
                            accessor,
                            row0,
                            chunk_row_bytes,
                            byte_offset,
                            cb_base + (idx % cb_depth) * window_bytes,
                            chunk_row_bytes,
                            tile_height);
                    }
                }
                {
                    DeviceZoneScopedN("RD-BARR");
                    noc_async_read_barrier();
                }
                {
                    DeviceZoneScopedN("RD-PUSH");
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            }
            return;
        }

        if constexpr (fanin_mode != 0) {
            // --- Refinement 2b: whole-page staged read (+ L1 redistribution) ------
            const uint32_t stage_page = get_arg_val<uint32_t>(7);
            const uint32_t stage_offset = get_arg_val<uint32_t>(8);
            // `blocks_per_core == 1` is a HOST gate and cannot be a static_assert (it
            // is not a compile-time arg), but the two runtime args that encode it are
            // right here. Without this a future host change that allowed a second block
            // would surface as a `cb_wait_front(cb_rm_input)` hang preceded by wrong
            // data -- and the single un-flow-controlled staging window would become a
            // genuine silent-corruption source (a mate would overwrite its cb_stage
            // with block 2's piece while this core is still pulling block 1).
            ASSERT(chunk_count == 1 && num_rows == tile_height);

            if constexpr (fanin_mode == 2) {
                // Measurement probe: the staged read alone, straight into the CB
                // window (piece_bytes == chunk_wt * tile_bytes, so it fills exactly
                // one window). No exchange => the tile layout is garbage; this row
                // exists only to price the read-side ceiling.
                cb_reserve_back(cb_rm_input, chunk_wt);
                if constexpr (!skip_dm) {
                    noc_async_read(
                        accessor.get_noc_addr(stage_page, stage_offset), get_write_ptr(cb_rm_input), piece_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_rm_input, chunk_wt);
                // NOC_CTRL is sticky ACROSS KERNEL LAUNCHES, so every exit from this
                // kernel owes the restore -- an early `return` that skips it leaks this
                // core's custom read VC into the next, unrelated program on this core.
                if constexpr (read_vc_spread) {
                    noc_async_read_one_packet_set_state<true>(
                        accessor.get_noc_addr(start_row), chunk_row_bytes, default_read_vc);
                }
                return;
            }

            constexpr uint32_t grp_h = group_size / grp_w;
            const uint32_t my_slot = get_arg_val<uint32_t>(9);
            tt_l1_ptr uint32_t* grp_x = (tt_l1_ptr uint32_t*)get_arg_addr(10);
            tt_l1_ptr uint32_t* grp_y = (tt_l1_ptr uint32_t*)get_arg_addr(10 + grp_w);

            // Phase 1 -- ONE contiguous read of this core's piece of one source page.
            const uint32_t stage_addr = get_write_ptr(cb_stage);
            if constexpr (!skip_dm) {
                noc_async_read(accessor.get_noc_addr(stage_page, stage_offset), stage_addr, piece_bytes);
            }
            noc_async_read_barrier();

            // Phase 2 -- ready handshake. Posted atomics: nothing has to be acked
            // back, and the payload they announce is this core's OWN L1, so there is
            // no write/atomic ordering hazard to close. Self is included through the
            // NoC (a local `*sem += 1` is not atomic against the 31 remote incs).
            const uint32_t sem_addr = get_semaphore(sem_fanin_id);
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            for (uint32_t r = 0; r < group_size; ++r) {
                noc_semaphore_inc<true>(get_noc_addr(grp_x[r % grp_w], grp_y[r / grp_w], sem_addr), 1);
            }
            noc_semaphore_wait_min(sem, group_size);

            // Phase 3 -- pull stick r out of group-mate r's staging buffer. `my_slot`
            // is this core's column slice inside the piece, identical on every mate.
            cb_reserve_back(cb_rm_input, chunk_wt);
            uint32_t l1_addr = get_write_ptr(cb_rm_input);
            const uint32_t mate_offset = stage_addr + my_slot * chunk_row_bytes;
            for (uint32_t r = 0; r < group_size; ++r) {
                if constexpr (!skip_dm) {
                    noc_async_read(
                        get_noc_addr(grp_x[r % grp_w], grp_y[r / grp_w], mate_offset), l1_addr, chunk_row_bytes);
                }
                l1_addr += chunk_row_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm_input, chunk_wt);
            if constexpr (read_vc_spread) {  // see the probe exit above
                noc_async_read_one_packet_set_state<true>(
                    accessor.get_noc_addr(start_row), chunk_row_bytes, default_read_vc);
            }
            return;
        }

        if constexpr (row_page_stride == 1 && prefetch_blocks == 2) {
            // --- lever B8: trid double-issue over the whole (chunk, block) run --
            // The sequence is flattened so the pipeline spans chunk boundaries too;
            // the order stays chunk-outer / tile-row-inner, which is what the
            // writer and compute both assume.
            constexpr uint32_t tile_bytes = get_tile_size(cb_rm_input);
            constexpr uint32_t window_bytes = chunk_wt * tile_bytes;
            const uint32_t blocks_per_chunk = num_rows / tile_height;
            const uint32_t total_blocks = chunk_count * blocks_per_chunk;
            // The FIFO write pointer starts at the CB base and advances exactly one
            // window per `cb_push_back(chunk_wt)`, wrapping after cb_depth pushes, so
            // window w is always base + w*window_bytes. Read BEFORE the first reserve
            // on purpose: the firmware re-runs
            // setup_local_cb_read_write_interfaces() at the top of every launch
            // (ncrisc.cc), so fifo_wr_ptr == fifo_addr here regardless of how many
            // pushes the PREVIOUS launch of this cached program made.
            const uint32_t cb_base = get_write_ptr(cb_rm_input);

            // Two windows free => the one block 0 lands in, and the one block 1
            // will land in before block 0 is published.
            cb_reserve_back(cb_rm_input, 2 * chunk_wt);
            noc_async_read_set_trid(trid_a);
            if constexpr (!skip_dm) {
                dataflow_kernel_lib::read_stick_rows_for_tilize<StickReadMode::Generic, 1>(
                    accessor,
                    start_row,
                    chunk_row_bytes,
                    chunk_start * chunk_row_bytes,
                    cb_base,
                    chunk_row_bytes,
                    tile_height);
            }

            for (uint32_t block = 0; block < total_blocks; ++block) {
                if (block + 1 < total_blocks) {
                    const uint32_t nxt = block + 1;
                    const uint32_t nc = nxt / blocks_per_chunk;
                    const uint32_t nr = nxt - nc * blocks_per_chunk;
                    noc_async_read_set_trid((nxt & 1u) ? trid_b : trid_a);
                    if constexpr (skip_dm) {
                        // Ablation: the payload goes, the address generation stays.
                        // The non-prefetched fallback below keeps its 32 accessor
                        // calls behind a volatile sink for exactly this reason, and
                        // Refinement 1 priced address-gen at ~437 ns of 3 609 ns on
                        // `d_tall_narrow` — dropping it here would bias every
                        // skip_dm A/B in this lever's favour by that whole term.
                        for (uint32_t row = 0; row < tile_height; ++row) {
                            volatile uint32_t sink = static_cast<uint32_t>(accessor.get_noc_addr(
                                start_row + nr * tile_height + row, (chunk_start + nc) * chunk_row_bytes));
                            (void)sink;
                        }
                    } else {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<StickReadMode::Generic, 1>(
                            accessor,
                            start_row + nr * tile_height,
                            chunk_row_bytes,
                            (chunk_start + nc) * chunk_row_bytes,
                            cb_base + (nxt % cb_depth) * window_bytes,
                            chunk_row_bytes,
                            tile_height);
                    }
                }
                noc_async_read_barrier_with_trid((block & 1u) ? trid_b : trid_a);
                cb_push_back(cb_rm_input, chunk_wt);
                if (block + 2 < total_blocks) {
                    // Guarantees window (block+2) % cb_depth carries no unpopped
                    // data before the next iteration issues into it.
                    cb_reserve_back(cb_rm_input, 2 * chunk_wt);
                }
            }
            // NOC_PACKET_TAG is sticky across kernel launches -- hand the cmd buf
            // back with the firmware's default tag.
            noc_async_read_set_trid(0);
            if constexpr (read_vc_spread) {
                noc_async_read_one_packet_set_state<true>(
                    accessor.get_noc_addr(start_row), chunk_row_bytes, default_read_vc);
            }
            return;
        }

        if constexpr (coalesce_rows || blocks_row_major || read_group > 1 || addr_probe) {
            // --- Refinement 3: one explicit (row-block, chunk) sequence ----------
            // `blocks_row_major` selects WHICH order (row-outer for the chunked
            // aliased output, chunk-outer otherwise — identical when either count
            // is 1), `coalesce_rows` selects HOW each block is read, and
            // `read_group` how many blocks share ONE barrier.
            //
            // read_group is lever B7 at a coarser granularity, and the measurement
            // that motivates it is the ablation of the `alias_out` crossover: its
            // read payload is 10 199 ns for 2.10 MB = 206 GB/s, i.e. already at the
            // best read rate this op has ever measured (the 1024 B generic path gets
            // 214), but its `sync_only` is 5 616 ns — 3 815 ns of which is the 256
            // `accessor.get_noc_addr` calls and the rest per-block CB/barrier cost.
            // A per-block `noc_async_read_barrier()` drains the read pipeline to
            // ZERO once per block, so with 8 blocks the DRAM round-trip latency is
            // exposed 8 times and neither the address generation of the next block
            // nor its reads can hide behind it. Grouping G blocks under one barrier
            // keeps 32*G reads in flight and exposes the drain total/G times. It
            // needs the same window arithmetic lever B8 uses (the FIFO write pointer
            // only advances on push, so the G windows this group writes into are
            // derived from the CB base) and a CB deep enough to hold them, which the
            // host guarantees (`depth >= read_group + 1`).
            const uint32_t blocks = num_rows / tile_height;
            const uint32_t total = blocks * chunk_count;
            constexpr uint32_t tile_bytes_in = get_tile_size(cb_rm_input);
            constexpr uint32_t window_bytes = chunk_wt * tile_bytes_in;
            // Read BEFORE the first reserve: `cb_reserve_back` does not move the
            // write pointer, and the firmware re-inits the CB interfaces per launch,
            // so this is the CB base even for a cached program.
            const uint32_t cb_base = get_write_ptr(cb_rm_input);
            for (uint32_t o = 0; o < total; o += read_group) {
                const uint32_t group = (total - o) < read_group ? (total - o) : read_group;
                cb_reserve_back(cb_rm_input, group * chunk_wt);
                for (uint32_t i = 0; i < group; ++i) {
                    const uint32_t idx = o + i;
                    const uint32_t block = blocks_row_major ? (idx / chunk_count) : (idx % blocks);
                    const uint32_t c = blocks_row_major ? (idx % chunk_count) : (idx / blocks);
                    const uint32_t row0 = start_row + block * tile_height;
                    const uint32_t byte_offset = (chunk_start + c) * chunk_row_bytes;
                    const uint32_t l1_addr = cb_base + (idx % cb_depth) * window_bytes;
                    if constexpr (coalesce_rows) {
                        // The chunk covers a whole source page (static_assert above),
                        // so `byte_offset` selects a page COLUMN and never an
                        // intra-page offset; rows row0..row0+31 are then 32
                        // consecutive pages of the SAME shard (the host gates
                        // shard_h % 32 == 0), i.e. one contiguous run in the owner's
                        // L1 — and the destination is contiguous as well, because the
                        // L1 row stride IS the page size.
                        const uint32_t page_col = byte_offset / source_page_bytes;
                        if constexpr (!skip_dm) {
                            noc_async_read(
                                accessor.get_noc_addr(row0 * row_page_stride + page_col),
                                l1_addr,
                                tile_height * chunk_row_bytes);
                        } else {
                            volatile uint32_t sink =
                                static_cast<uint32_t>(accessor.get_noc_addr(row0 * row_page_stride + page_col));
                            (void)sink;
                        }
                        // Checked, not assumed: the run this folds into one
                        // transaction must really be contiguous in the source
                        // (watcher builds only).
                        ASSERT(
                            accessor.get_noc_addr((row0 + tile_height - 1) * row_page_stride + page_col) ==
                            accessor.get_noc_addr(row0 * row_page_stride + page_col) +
                                (tile_height - 1) * chunk_row_bytes);
                    } else if constexpr (addr_probe) {
                        // Timing probe: ONE accessor call per block, then a running
                        // increment. Wrong bytes on purpose (rows of an interleaved
                        // tensor are in different banks) -- bench-only.
                        const uint64_t base = accessor.get_noc_addr(row0, byte_offset);
                        for (uint32_t row = 0; row < tile_height; ++row) {
                            if constexpr (!skip_dm) {
                                noc_async_read(base, l1_addr + row * chunk_row_bytes, chunk_row_bytes);
                            }
                        }
                    } else if constexpr (!skip_dm) {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<read_mode, 1>(
                            accessor, row0, chunk_row_bytes, byte_offset, l1_addr, chunk_row_bytes, tile_height);
                    } else {
                        for (uint32_t row = 0; row < tile_height; ++row) {
                            volatile uint32_t sink =
                                static_cast<uint32_t>(accessor.get_noc_addr(row0 + row, byte_offset));
                            (void)sink;
                        }
                    }
                }
                // ONE barrier for the whole group (lever B7 at group granularity).
                noc_async_read_barrier();
                // ONE push PER WINDOW, not one push of `group * chunk_wt`. A single
                // push may not straddle the end of the FIFO -- `cb_push_back` only
                // handles the exact-hit wrap ("no other wrap is legal", it advances
                // `fifo_wr_ptr` and then subtracts `fifo_size` only on equality,
                // dataflow_api.h:213-222) -- and with `depth == group + 1` a group DOES
                // straddle it every other iteration. Caught by the lightweight
                // `ASSERT(fifo_wr_ptr <= fifo_limit)` under `--dev` (an ebreak, i.e. a
                // HANG) while the default build silently left the pointer past the
                // limit. Per-window pushes are each contiguous by construction, and
                // publishing them back-to-back after the shared barrier keeps the
                // lever's meaning (one barrier per group) intact.
                for (uint32_t i = 0; i < group; ++i) {
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            }
            if constexpr (read_vc_spread) {  // NOC_CTRL is sticky across launches
                noc_async_read_one_packet_set_state<true>(
                    accessor.get_noc_addr(start_row), chunk_row_bytes, default_read_vc);
            }
            return;
        }

        for (uint32_t c = 0; c < chunk_count; ++c) {
            const uint32_t byte_offset = (chunk_start + c) * chunk_row_bytes;

            if constexpr (row_page_stride == 1 && stagger && !split_read) {
                // --- Refinement 2b: rotated issue order (see the header) ---------
                // Two helper calls == the two row runs [rot, 32) and [0, rot). The
                // L1 destination still follows the row index, so the block the
                // compute kernel sees is byte-identical to the unrotated one.
                const uint32_t rot = get_arg_val<uint32_t>(6);
                for (uint32_t block = 0; block < num_rows / tile_height; ++block) {
                    const uint32_t row0 = start_row + block * tile_height;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    const uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    if constexpr (!skip_dm) {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<StickReadMode::Generic, 1>(
                            accessor,
                            row0 + rot,
                            chunk_row_bytes,
                            byte_offset,
                            l1_addr + rot * chunk_row_bytes,
                            chunk_row_bytes,
                            tile_height - rot);
                        if (rot != 0) {
                            dataflow_kernel_lib::read_stick_rows_for_tilize<StickReadMode::Generic, 1>(
                                accessor, row0, chunk_row_bytes, byte_offset, l1_addr, chunk_row_bytes, rot);
                        }
                    } else {
                        for (uint32_t row = 0; row < tile_height; ++row) {
                            volatile uint32_t sink =
                                static_cast<uint32_t>(accessor.get_noc_addr(row0 + row, byte_offset));
                            (void)sink;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            } else if constexpr (row_page_stride == 1 && !split_read && !skip_dm) {
                dataflow_kernel_lib::
                    read_sticks_for_tilize<cb_rm_input, dataflow_kernel_lib::TilizeGranularity::TILE, read_mode>(
                        accessor, num_rows, chunk_row_bytes, start_row, byte_offset);
            } else if constexpr (row_page_stride == 1 && split_read) {
                // Lever C7. The CB dance stays here (single producer); the row
                // band is split with BRISC by bank group inside the helper.
                volatile tt_l1_ptr uint32_t* sem_reserve =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_reserve_id));
                volatile tt_l1_ptr uint32_t* sem_done =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done_id));
                const uint32_t blocks = num_rows / tile_height;

                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t first_page = start_row + block * tile_height;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    const uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    // The window is free: hand it to BRISC. Sequence numbers are
                    // per (chunk, block) so they stay monotonic across chunks.
                    const uint32_t seq = c * blocks + block + 1;
                    noc_semaphore_set(sem_reserve, seq);

                    if constexpr (!skip_dm) {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<read_mode, 2>(
                            accessor,
                            first_page,
                            chunk_row_bytes,
                            byte_offset,
                            l1_addr,
                            chunk_row_bytes,
                            tile_height,
                            /*split_id=*/0);
                    }
                    noc_async_read_barrier();
                    noc_semaphore_wait_min(sem_done, seq);
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            } else {
                // A chunk never straddles a source page (host guarantees
                // chunk_row_bytes divides source_page_bytes), so the whole
                // chunk lives in one page at a fixed intra-page offset.
                const uint32_t page_col = byte_offset / source_page_bytes;
                const uint32_t offset_in_page = byte_offset - page_col * source_page_bytes;
                const uint32_t blocks = num_rows / tile_height;

                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t row0 = start_row + block * tile_height;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    for (uint32_t row = 0; row < tile_height; ++row) {
                        const uint64_t noc_addr =
                            accessor.get_noc_addr((row0 + row) * row_page_stride + page_col, offset_in_page);
                        if constexpr (skip_dm) {
                            // Ablation: keep the address-gen observable so dead-code
                            // elimination cannot delete the loop being timed.
                            volatile uint32_t sink = static_cast<uint32_t>(noc_addr);
                            (void)sink;
                        } else {
                            noc_async_read(noc_addr, l1_addr, chunk_row_bytes);
                        }
                        l1_addr += chunk_row_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            }
        }

        if constexpr (read_vc_spread) {
            // Restore the firmware default before exiting -- NOC_CTRL survives the
            // kernel launch and the next program on this core will not re-set it.
            noc_async_read_one_packet_set_state<true>(
                accessor.get_noc_addr(start_row), chunk_row_bytes, default_read_vc);
        }
    }
}
