// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize writer (BRISC / NoC1).
//
//   alias_mode == 1  (Path B, and Refinement 3's `alias_out` — zero-copy WRITE)
//       cb_tiled_output is aliased onto the resident L1 TILE shard, so compute
//       has already written the bytes to their final address. One
//       cb_wait_front / cb_pop_front drains the shard; no NoC traffic.
//       Refinement 3 reaches this branch for an interleaved RM input with a sharded
//       TILE output: the work split gives this core exactly its own output shard
//       (`_one_sided_shard_split`), so CB page k IS shard tile k and the tilize LLK
//       packs straight into its final address. That deletes 6 786 ns of 19 150 on
//       `g_dram_to_sharded` (35 % of the runtime — the one-sided DM ablation) and
//       leaves BRISC free for lever C7 below.
//
//   alias_mode == 0  (Path A / C)
//       Whole-TILE-page writes through the output TensorAccessor, chunk_wt
//       writes per barrier (lever B7).
//
//   stagger == 1  (Refinement 2b — per-core write-order rotation)
//       Output tile page p lives in DRAM bank `p % NUM_DRAM_BANKS`, and every core
//       writes its `chunk_wt` pages in ascending order, so the instantaneous bank
//       demand is clustered on a few banks (with chunk_wt = 8 over 12 banks the 64
//       cores only ever start on 3 distinct banks). Rotating each core's write order
//       by `col_rot` spreads step 0 across the banks. Pure index permutation: same
//       pages, same size, same count, same CB bookkeeping.
//
//   split_read == 1  (lever C7 — this kernel also READS)
//       With one chunk-block per core BRISC would sit in cb_wait_front for the
//       whole read window (~1.5-3 us) and then work for ~1.4 us, so its NoC
//       issue capacity is the only structurally idle resource on the core. It
//       therefore takes half of each block's 32 stick reads before waiting for
//       that block's tilized output. NCRISC stays the ONLY producer of
//       cb_rm_input (single-producer rule): it reserves the window and hands it
//       over through sem_reserve, this kernel writes into it and answers on
//       sem_done.
//
//       Refinement 3 lifts the old `depth == 1` restriction. This kernel never
//       reserves or pushes cb_rm_input, so its copy of the CB write pointer never
//       leaves the base — which used to mean `get_write_ptr` was only the RIGHT
//       window at depth 1. The window is now DERIVED instead:
//       `cb_base + (block % cb_depth) * window_bytes`, which is exactly where the
//       FIFO write pointer stands after `block` pushes of chunk_wt pages (the same
//       identity lever B8 relies on, and the reserve NCRISC already did is what
//       guarantees the window is free). That matters on the `alias_out` crossover:
//       there the only thing left to overlap the reads with is the tilize LLK, so
//       C7 must not cost the depth-2 reader/compute pipeline to get its second
//       issuer.
//
// RAW-API NOTE (helper substitution, deliberate): there is no kernel_lib
// dataflow helper that moves TILE pages from a CB to a TensorAccessor-addressed
// buffer. The only write helper, dataflow_kernel_lib::write_sticks_after_untilize
// (tilize_helpers_dataflow.hpp), writes ROW_MAJOR *sticks* — its inner loop
// issues one noc_async_write of row_bytes per row and advances the L1 pointer by
// padded_row_bytes, i.e. it de-interleaves a tile back into 32 sticks. It is the
// untilize partner, the wrong direction; using it here would write tile bytes to
// stick addresses and destroy the layout. The read half above DOES use a helper
// (dataflow_kernel_lib::read_stick_rows_for_tilize).
//
// The iteration order MUST match the reader's: chunk-outer, tile-row-inner.
// read_sticks_for_tilize loops over tile-row blocks internally, so the chunk
// loop is the caller's outer loop. Reversing it keeps every CB count balanced
// and silently transposes the output blocks — and with split_read on it would
// also pair the wrong source rows with the reserved window.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_input = 0;
    constexpr uint32_t cb_tiled_output = 16;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t alias_mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t wt = get_compile_time_arg_val(3);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(4);
    // Perf-ablation only (TILIZE_SKIP_DM=1) — see the reader for the contract.
    constexpr uint32_t skip_dm = get_compile_time_arg_val(5);
    constexpr uint32_t split_read = get_compile_time_arg_val(6);  // lever C7
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t stateful_read = get_compile_time_arg_val(8);  // lever B13
    constexpr uint32_t sem_reserve_id = get_compile_time_arg_val(9);
    constexpr uint32_t sem_done_id = get_compile_time_arg_val(10);
    constexpr uint32_t vc_spread = get_compile_time_arg_val(11);  // lever B10 (bitmask)
    constexpr bool write_vc_spread = (vc_spread & 2u) != 0;       // bit 1 == spread the writes
    constexpr uint32_t stagger = get_compile_time_arg_val(12);    // Refinement 2b
    // Refinement 3: the input CB's window count, so the C7 read half can derive the
    // window NCRISC reserved at ANY depth (see the header).
    constexpr uint32_t cb_in_depth = get_compile_time_arg_val(13);
    constexpr auto dst_args = TensorAccessorArgs<14>();
    // Declared unconditionally (never inside `if constexpr`) so the CT arg offsets
    // are the same in both configurations; only used when split_read.
    [[maybe_unused]] constexpr auto src_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    using dataflow_kernel_lib::StickReadMode;
    constexpr StickReadMode read_mode = stateful_read ? StickReadMode::Stateful : StickReadMode::Generic;
    // The write rotation and the C7 read half touch disjoint CBs and disjoint
    // semaphores, so the combination is harmless today -- but the host never produces
    // it, and this is the compile-time tripwire if a future gate loosens.
    static_assert(!stagger || !split_read, "the write-order rotation and the C7 split reader are not paired");

    if constexpr (alias_mode) {
        if constexpr (split_read) {
            // --- lever C7 on the `alias_out` crossover -------------------------
            // With the output aliased this kernel has NO writes, so its whole job
            // is the other half of each block's stick reads. Same two-semaphore
            // hand-off as the generic branch below (NCRISC stays the only producer
            // of cb_rm_input), just hoisted ahead of the single output drain —
            // there is no per-block output wait to interleave with.
            const uint32_t row_start = get_arg_val<uint32_t>(1);
            const uint32_t row_count = get_arg_val<uint32_t>(2);
            const uint32_t chunk_start = get_arg_val<uint32_t>(3);
            const uint32_t src_addr = get_arg_val<uint32_t>(5);
            const auto src_accessor = TensorAccessor(src_args, src_addr);
            volatile tt_l1_ptr uint32_t* sem_reserve =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_reserve_id));
            volatile tt_l1_ptr uint32_t* sem_done =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done_id));
            const uint32_t cb_in_base = get_write_ptr(cb_rm_input);
            constexpr uint32_t cb_in_window_bytes = chunk_wt * get_tile_size(cb_rm_input);

            for (uint32_t r = 0; r < row_count; ++r) {
                const uint32_t seq = r + 1;
                noc_semaphore_wait_min(sem_reserve, seq);
                if constexpr (!skip_dm) {
                    dataflow_kernel_lib::read_stick_rows_for_tilize<read_mode, 2>(
                        src_accessor,
                        (row_start + r) * tile_height,
                        chunk_row_bytes,
                        chunk_start * chunk_row_bytes,
                        cb_in_base + ((seq - 1) % cb_in_depth) * cb_in_window_bytes,
                        chunk_row_bytes,
                        tile_height,
                        /*split_id=*/1);
                }
                noc_async_read_barrier();
                noc_semaphore_set(sem_done, seq);
            }
        }
        cb_wait_front(cb_tiled_output, shard_tiles);
        cb_pop_front(cb_tiled_output, shard_tiles);
        return;
    } else {
        const uint32_t dst_addr = get_arg_val<uint32_t>(0);
        const uint32_t row_start = get_arg_val<uint32_t>(1);
        const uint32_t row_count = get_arg_val<uint32_t>(2);
        const uint32_t chunk_start = get_arg_val<uint32_t>(3);
        const uint32_t chunk_count = get_arg_val<uint32_t>(4);

        const auto accessor = TensorAccessor(dst_args, dst_addr);
        // Lever B10 (per-writer static unicast VC). Unlike the read path this needs
        // no sticky-register dance and no restore: ncrisc_noc_fast_write writes
        // NOC_CTRL (and therefore NOC_CMD_STATIC_VC) on EVERY call, so the vc
        // argument of noc_async_write is live in DM_DEDICATED_NOC.
        // Only read when the lever is on: passing a *runtime* vc unconditionally
        // stops the compiler folding NOC_CMD_STATIC_VC(vc) into the constant
        // NOC_CTRL word, i.e. it would make the lever cost something even when off.
        [[maybe_unused]] const uint32_t write_vc = write_vc_spread ? get_arg_val<uint32_t>(6) : NOC_UNICAST_WRITE_VC;
        // Refinement 2b: this core's rotation of the in-block write order. Read as a
        // constant 0 when the lever is off, so the loop below folds back to `k`.
        const uint32_t col_rot = stagger ? get_arg_val<uint32_t>(7) : 0;
        // Lever C7's window base. Read BEFORE the loop and never updated: this
        // kernel is not a producer of cb_rm_input, so its CB write pointer stays at
        // the base for the whole launch (the firmware re-runs
        // setup_local_cb_read_write_interfaces() per launch, so this is the base even
        // for a cached program whose previous launch pushed).
        [[maybe_unused]] const uint32_t cb_in_base = split_read ? get_write_ptr(cb_rm_input) : 0;
        constexpr uint32_t cb_in_window_bytes = chunk_wt * get_tile_size(cb_rm_input);

        for (uint32_t c = 0; c < chunk_count; ++c) {
            const uint32_t col0 = (chunk_start + c) * chunk_wt;
            for (uint32_t r = 0; r < row_count; ++r) {
                if constexpr (split_read) {
                    // --- lever C7: this block's other half of the stick reads ---
                    const uint32_t src_addr = get_arg_val<uint32_t>(5);
                    const auto src_accessor = TensorAccessor(src_args, src_addr);
                    volatile tt_l1_ptr uint32_t* sem_reserve =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_reserve_id));
                    volatile tt_l1_ptr uint32_t* sem_done =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done_id));

                    const uint32_t seq = c * row_count + r + 1;
                    noc_semaphore_wait_min(sem_reserve, seq);
                    // The window NCRISC reserved for block `seq - 1`. At depth 1 this
                    // is the base (the old `get_write_ptr` value); past that it
                    // rotates, exactly like the FIFO write pointer does.
                    const uint32_t window = cb_in_base + ((seq - 1) % cb_in_depth) * cb_in_window_bytes;
                    if constexpr (!skip_dm) {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<read_mode, 2>(
                            src_accessor,
                            (row_start + r) * tile_height,
                            chunk_row_bytes,
                            (chunk_start + c) * chunk_row_bytes,
                            window,
                            chunk_row_bytes,
                            tile_height,
                            /*split_id=*/1);
                    }
                    noc_async_read_barrier();
                    noc_semaphore_set(sem_done, seq);
                }

                const uint32_t base_page = (row_start + r) * wt + col0;

                cb_wait_front(cb_tiled_output, chunk_wt);
                const uint32_t l1_addr = get_read_ptr(cb_tiled_output);
                for (uint32_t i = 0; i < chunk_wt; ++i) {
                    // `k` is the tile inside the block; the rotation only changes the
                    // ISSUE ORDER, never the (page, L1 address) pairing.
                    const uint32_t k = (i + col_rot) < chunk_wt ? (i + col_rot) : (i + col_rot - chunk_wt);
                    const uint64_t noc_addr = accessor.get_noc_addr(base_page + k);
                    if constexpr (skip_dm) {
                        volatile uint32_t sink = static_cast<uint32_t>(noc_addr);
                        (void)sink;
                    } else if constexpr (write_vc_spread) {
                        noc_async_write(l1_addr + k * tile_bytes, noc_addr, tile_bytes, noc_index, write_vc);
                    } else {
                        noc_async_write(l1_addr + k * tile_bytes, noc_addr, tile_bytes);
                    }
                }
                // Recycling the CB pages only requires the writes to have DEPARTED
                // (the data read out of L1), not to have been acked by the
                // destination — that is exactly noc_async_writes_flushed
                // (dataflow_api.h:1802 "wait for ... calls to depart, but will not
                // wait for them to complete"). A full barrier per block would idle
                // BRISC for the round-trip latency of the last tile of every block.
                // One barrier after the loop still guarantees completion before the
                // kernel ends.
                noc_async_writes_flushed();
                cb_pop_front(cb_tiled_output, chunk_wt);
            }
        }
        noc_async_write_barrier();
    }
}
