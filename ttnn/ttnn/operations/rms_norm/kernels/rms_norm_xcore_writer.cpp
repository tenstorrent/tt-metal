// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-core W-split transport kernel for rms_norm (op_design.md §5).
//
// This dataflow kernel owns the cross-core stat traffic (the output shard itself
// is a zero-copy CB, so there is no DRAM write). One fully-synchronous round per
// tile-row, Pattern A (gather -> master combine -> broadcast), all-unicast so it
// is topology-agnostic (WIDTH auto-shard groups can be non-rectangular on the
// 11-wide grid; NoC-mcast/two-stage topology is the R6 perf lever). Three
// MONOTONE counter semaphores (no reset -> no clobber race, §9):
//   SEM_GATHER : worker -> master, "my partial for this row landed".
//   SEM_BCAST  : master -> worker, "the finalized 1/RMS for this row landed".
//   SEM_DONE   : worker -> master, "I reserved my broadcast slot (prev row
//                consumed via CB back-pressure) -> master may overwrite it".
//
// Cross-core-written CBs use FIXED base addresses (same L1 offset on every core):
//   cb_gather   depth K -> master pops K each round, wrapping the fifo to base.
//   cb_stat_global depth 1 -> always base.
// so a remote writer targets `get_write_ptr(cb) [+ slot]` computed locally.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_shard_out = 9;  // RM output: zero-copy alias of the resident RM W-slice (stick pages)
constexpr uint32_t cb_gather = 5;
constexpr uint32_t cb_stat_handoff = 6;
constexpr uint32_t cb_stat_global = 7;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_sticks = 17;  // RM output: tile-padded sticks (compute untilize) -> loopback shard
constexpr uint32_t cb_stat_local = 25;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(0);
    constexpr uint32_t SEM_BCAST = get_compile_time_arg_val(1);
    constexpr uint32_t SEM_DONE = get_compile_time_arg_val(2);
    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t stat_bytes = get_compile_time_arg_val(5);  // fp32 tile bytes
    // OUT_TO_DRAM (Refinement 4a, logical wide-interleaved / decode W-split): the
    // output is INTERLEAVED, not a zero-copy sharded CB, so after the per-tile-row
    // stat round this writer drains compute's cb_out (per_w_t tiles) and writes the
    // vwt valid W-slice tiles to DRAM (tile_id = t*Wt + w_tile_start + w). When
    // OUT_TO_DRAM=0 the output is a zero-copy sharded CB and compute's pack finalizes
    // it in place (writer does not touch cb_out).
    constexpr bool OUT_TO_DRAM = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(8);
    constexpr uint32_t out_page = get_compile_time_arg_val(9);
    // IS_RM (Refinement 4b): the resident output W-slice is ROW-MAJOR. After each
    // tile-row's stat round this writer loopback-copies compute's untilized cb_out_sticks
    // (tile-padded) into the resident RM output shard (cb_shard_out), valid columns only.
    constexpr bool IS_RM = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t ELEM = get_compile_time_arg_val(11);               // element bytes (RM loopback math)
    constexpr uint32_t SHARD_STICK_BYTES = get_compile_time_arg_val(12);  // resident RM shard stick stride

    constexpr auto out_args = TensorAccessorArgs<13>();

    // DRAM-write args first (fixed-position); the variable-length worker coords follow.
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t vwt = get_arg_val<uint32_t>(2);
    const uint32_t is_master = get_arg_val<uint32_t>(3);
    const uint32_t slice_index = get_arg_val<uint32_t>(4);
    const uint32_t master_vx = get_arg_val<uint32_t>(5);
    const uint32_t master_vy = get_arg_val<uint32_t>(6);
    const uint32_t valid_cols = get_arg_val<uint32_t>(7);        // IS_RM: valid output columns
    const uint32_t valid_rows_total = get_arg_val<uint32_t>(8);  // IS_RM: valid rows in this core's shard
    const uint32_t phase = get_arg_val<uint32_t>(9);             // IS_RM: w_offset % 32 (leading tile offset)
    const uint32_t num_workers = get_arg_val<uint32_t>(10);      // master: K-1, worker: 0
    // Refinement 6 (collective-topology lever): when this group's cores form a GAP-FREE
    // rectangle (host sets use_mcast=1 iff bounding-box area == group size), the master
    // broadcasts the finalized 1/RMS with ONE noc_async_write_multicast + ONE
    // noc_semaphore_set_multicast to the rectangle instead of K-1 serial unicast writes +
    // K-1 sem incs — a K-independent broadcast. Ragged groups (the logical decode W-split's
    // first-K row-major cores on the 11-wide grid, or WIDTH auto-shard groups wrapping a
    // partial row) get use_mcast=0 and keep the topology-agnostic all-unicast fallback.
    // rect corners are VIRTUAL NoC coords; the master is the low corner of the rectangle.
    const uint32_t use_mcast = get_arg_val<uint32_t>(11);
    const uint32_t rect_xlo = get_arg_val<uint32_t>(12);
    const uint32_t rect_ylo = get_arg_val<uint32_t>(13);
    const uint32_t rect_xhi = get_arg_val<uint32_t>(14);
    const uint32_t rect_yhi = get_arg_val<uint32_t>(15);
    // master only: worker virtual coords follow as [vx, vy] * num_workers (used by the
    // all-unicast fallback; unused on the mcast path but always emitted by the host).
    constexpr uint32_t WORKER_COORDS_BASE = 16;

    volatile tt_l1_ptr uint32_t* sem_gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));
    volatile tt_l1_ptr uint32_t* sem_bcast = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BCAST));
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE));

    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page);

    for (uint32_t t = 0; t < HT_LOCAL; ++t) {
        cb_wait_front(cb_stat_local, 1);  // compute produced this tile-row's local partial
        const uint32_t local_src = get_read_ptr(cb_stat_local);

        if (is_master) {
            // ---- gather: own partial into slot 0, wait workers' slots ----
            cb_reserve_back(cb_gather, K);  // blocks until compute popped last round's gather
            const uint32_t gather_base = get_write_ptr(cb_gather);
            noc_async_read(get_noc_addr(master_vx, master_vy, local_src), gather_base, stat_bytes);  // loopback own
            noc_async_read_barrier();
            cb_pop_front(cb_stat_local, 1);
            noc_semaphore_wait_min(sem_gather, (t + 1) * (K - 1));  // all worker partials landed
            cb_push_back(cb_gather, K);                             // -> master compute combine

            // ---- broadcast: wait combined 1/RMS, push to every group member ----
            cb_wait_front(cb_stat_handoff, 1);
            const uint32_t handoff_src = get_read_ptr(cb_stat_handoff);
            cb_reserve_back(cb_stat_global, 1);  // blocks until compute popped last round's rstd
            const uint32_t global_dst = get_write_ptr(cb_stat_global);
            noc_semaphore_wait_min(sem_done, (t + 1) * (K - 1));  // all workers ready to receive
            noc_async_read(get_noc_addr(master_vx, master_vy, handoff_src), global_dst, stat_bytes);  // loopback own
            if (use_mcast) {
                // ONE mcast of the finalized 1/RMS to the group rectangle (excl. self;
                // num_dests = num_workers = K-1). Corners in routing order for this NoC:
                // NoC0 walks low->high, NoC1 high->low (master is the low corner).
                const uint64_t mcast_dst =
                    (noc_index == 1) ? get_noc_multicast_addr(rect_xhi, rect_yhi, rect_xlo, rect_ylo, global_dst)
                                     : get_noc_multicast_addr(rect_xlo, rect_ylo, rect_xhi, rect_yhi, global_dst);
                noc_async_write_multicast(handoff_src, mcast_dst, stat_bytes, num_workers);
            } else {
                for (uint32_t w = 0; w < num_workers; ++w) {
                    const uint32_t wx = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w);
                    const uint32_t wy = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w + 1);
                    noc_async_write(handoff_src, get_noc_addr(wx, wy, global_dst), stat_bytes);
                }
            }
            noc_async_write_barrier();  // data flushed before the ready signal
            noc_async_read_barrier();
            cb_push_back(cb_stat_global, 1);  // -> master compute pass 2
            cb_pop_front(cb_stat_handoff, 1);
            if (use_mcast) {
                // Signal all workers with ONE mcast semaphore set. Monotone: set the value
                // to (t+1) each round (matches the workers' noc_semaphore_wait_min(t+1)); the
                // set replaces the K-1 per-worker atomic incs. The existing SEM_DONE
                // back-pressure bounds the master to one round ahead, so the depth-1
                // cb_stat_global is consumed before the next set overwrites it (no race).
                noc_semaphore_set(sem_bcast, t + 1);
                const uint64_t mcast_sem =
                    (noc_index == 1)
                        ? get_noc_multicast_addr(rect_xhi, rect_yhi, rect_xlo, rect_ylo, get_semaphore(SEM_BCAST))
                        : get_noc_multicast_addr(rect_xlo, rect_ylo, rect_xhi, rect_yhi, get_semaphore(SEM_BCAST));
                noc_semaphore_set_multicast(get_semaphore(SEM_BCAST), mcast_sem, num_workers);
                noc_async_write_barrier();  // flush the mcast signal (non-posted write)
            } else {
                for (uint32_t w = 0; w < num_workers; ++w) {
                    const uint32_t wx = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w);
                    const uint32_t wy = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w + 1);
                    noc_semaphore_inc(get_noc_addr(wx, wy, get_semaphore(SEM_BCAST)), 1);
                }
            }
        } else {
            // ---- worker: reserve receive slot (prev consumed), signal ready, gather ----
            cb_reserve_back(cb_stat_global, 1);  // blocks until compute popped last round's rstd
            noc_semaphore_inc(get_noc_addr(master_vx, master_vy, get_semaphore(SEM_DONE)), 1);  // ready to receive
            // master's cb_gather base == this core's cb_gather base (uniform CB offset, empty->base).
            const uint32_t master_gather_base = get_write_ptr(cb_gather);
            noc_async_write(
                local_src,
                get_noc_addr(master_vx, master_vy, master_gather_base + slice_index * stat_bytes),
                stat_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(master_vx, master_vy, get_semaphore(SEM_GATHER)), 1);  // partial landed
            cb_pop_front(cb_stat_local, 1);
            noc_semaphore_wait_min(sem_bcast, t + 1);  // master broadcast my 1/RMS
            cb_push_back(cb_stat_global, 1);           // -> worker compute pass 2 (data already at the slot)
        }

        // ---- logical W-split: drain this tile-row's compute output to DRAM ----
        // Compute pushes PER_W_T tiles per tile-row (pass 2 loops the whole slice);
        // write only the vwt valid W-slice tiles to interleaved DRAM (tile_id =
        // t*Wt + w_tile_start + w), pop the full PER_W_T to keep the CB balanced.
        if constexpr (OUT_TO_DRAM) {
            cb_wait_front(cb_out, PER_W_T);
            uint32_t l1 = get_read_ptr(cb_out);
            const uint32_t tile_bytes = get_tile_size(cb_out);
            for (uint32_t w = 0; w < vwt; ++w) {
                noc_async_write_tile(t * Wt + w_tile_start + w, out_accessor, l1);
                l1 += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, PER_W_T);
        }

        // ---- RM output (Refinement 4b): loopback compute's untilized sticks -> shard ----
        // Compute untilized PER_W_T tile-padded stick tiles into cb_out_sticks. Copy the
        // valid columns of each valid row into the resident RM output shard (cb_shard_out,
        // contiguous sticks at SHARD_STICK_BYTES stride); the sub-tile pad columns / pad
        // rows are tensor-padding (discarded on read-back), so they are not written.
        if constexpr (IS_RM) {
            constexpr uint32_t PADDED_ROW_BYTES = PER_W_T * TILE_DIM * ELEM;
            uint32_t valid_rows = (valid_rows_total > t * TILE_DIM) ? (valid_rows_total - t * TILE_DIM) : 0;
            if (valid_rows > TILE_DIM) {
                valid_rows = TILE_DIM;
            }
            cb_wait_front(cb_out_sticks, PER_W_T);
            const uint32_t src = get_read_ptr(cb_out_sticks) + phase * ELEM;  // valid columns start at `phase`
            const uint32_t shard_base = get_write_ptr(cb_shard_out);
            const uint32_t vc_bytes = valid_cols * ELEM;
            for (uint32_t s = 0; s < valid_rows; ++s) {
                const uint32_t dst = shard_base + (t * TILE_DIM + s) * SHARD_STICK_BYTES;
                noc_async_write(
                    src + s * PADDED_ROW_BYTES, get_noc_addr(my_x[noc_index], my_y[noc_index], dst), vc_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out_sticks, PER_W_T);
        }
    }
    // Flush the semaphore-inc atomics (SEM_GATHER/SEM_DONE from workers, SEM_BCAST
    // from the master) before the kernel exits — else the target may not observe
    // the final round's signal and the exit "atomics flushed" assert trips.
    noc_async_atomic_barrier();
}
