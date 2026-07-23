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
    // C_ROWS (Refinement 6a lever 1): number of tile-rows batched into ONE cross-core round.
    // Rounds = ceil(HT_LOCAL / C_ROWS); the gather/broadcast move C_this tiles per core per
    // round. C_ROWS=1 reduces to the R4 per-tile-row round. Monotone counter semaphores count
    // per ROUND (each worker incs SEM_GATHER/SEM_DONE once/round; master sets SEM_BCAST=r+1).
    constexpr uint32_t C_ROWS = get_compile_time_arg_val(13);
    // Stat-tile compaction (Refinement 6b lever 1). The cross-core stat is a REDUCE_ROW
    // result: only COLUMN 0 is ever consumed (the master fold is element-wise; pass-2 reads
    // it via BroadcastDim::Col). In an fp32 tile column 0 lives entirely in faces 0 (rows
    // 0-15) and 2 (rows 16-31) — i.e. byte ranges [0,1024) and [2048,3072). The gather
    // therefore only needs to move those bytes; the untransferred faces (1 always, 3 in the
    // 2-run form) leave STALE L1 that the fold sums-then-ignores, so the output is
    // numerically BYTE-IDENTICAL. The slot STRIDE stays a full fp32 tile (stat_bytes) — only
    // the moved bytes shrink. Configured as up to two contiguous runs so the host can select:
    //   full  (byte-identical R6a): (0, stat_bytes, _, 0)
    //   faces 0-2 contiguous 3 KB : (0, 3072, _, 0)
    //   faces 0 & 2 only 2 KB     : (0, 1024, 2048, 1024)
    constexpr uint32_t G_OFF0 = get_compile_time_arg_val(14);
    constexpr uint32_t G_LEN0 = get_compile_time_arg_val(15);
    constexpr uint32_t G_OFF1 = get_compile_time_arg_val(16);
    constexpr uint32_t G_LEN1 = get_compile_time_arg_val(17);

    constexpr auto out_args = TensorAccessorArgs<18>();

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
    // Refinement 6 collective-topology lever + Refinement 6a lever 2 (gap-aware mcast):
    // the master broadcasts the finalized 1/RMS with a NoC multicast instead of K-1 serial
    // unicast writes. A group whose VIRTUAL coords form one gap-free rectangle mcasts in ONE
    // segment (R6). Groups that straddle the Blackhole DRAM columns (virtual x=8,9) split
    // into up to TWO contiguous virtual runs ([xlo..7] + [10..xhi]), each a rectangle the
    // master mcasts to separately (R6a) — this unblocks the 8-wide WIDTH/BLOCK groups a naive
    // rectangle mcast would fault on. Truly ragged groups (logical decode; WIDTH auto-shard
    // wrapping a partial row) set n_mcast_seg=0 and keep the topology-agnostic all-unicast
    // fallback. Per segment: (xlo, ylo, xhi, yhi, ndests) VIRTUAL corners + destination count
    // (the sender master is auto-excluded from the segment it sits in, so its ndests is
    // seg_members-1; the other segment's ndests is its full member count).
    const uint32_t n_mcast_seg = get_arg_val<uint32_t>(11);
    const uint32_t seg0_xlo = get_arg_val<uint32_t>(12);
    const uint32_t seg0_ylo = get_arg_val<uint32_t>(13);
    const uint32_t seg0_xhi = get_arg_val<uint32_t>(14);
    const uint32_t seg0_yhi = get_arg_val<uint32_t>(15);
    const uint32_t seg0_nd = get_arg_val<uint32_t>(16);
    const uint32_t seg1_xlo = get_arg_val<uint32_t>(17);
    const uint32_t seg1_ylo = get_arg_val<uint32_t>(18);
    const uint32_t seg1_xhi = get_arg_val<uint32_t>(19);
    const uint32_t seg1_yhi = get_arg_val<uint32_t>(20);
    const uint32_t seg1_nd = get_arg_val<uint32_t>(21);
    // master only: worker virtual coords follow as [vx, vy] * num_workers (used by the
    // all-unicast fallback; unused on the mcast path but always emitted by the host).
    constexpr uint32_t WORKER_COORDS_BASE = 22;

    volatile tt_l1_ptr uint32_t* sem_gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));
    volatile tt_l1_ptr uint32_t* sem_bcast = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BCAST));
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE));

    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page);

    const uint32_t num_rounds = (HT_LOCAL + C_ROWS - 1) / C_ROWS;
    for (uint32_t r = 0; r < num_rounds; ++r) {
        const uint32_t base_t = r * C_ROWS;
        uint32_t C_this = HT_LOCAL - base_t;
        if (C_this > C_ROWS) {
            C_this = C_ROWS;
        }
        const uint32_t bcast_bytes = C_this * stat_bytes;

        // ---- cross-core stat round for the C_this tile-rows ----
        cb_wait_front(cb_stat_local, C_this);  // compute produced C_this local partials
        const uint32_t local_src = get_read_ptr(cb_stat_local);

        if (is_master) {
            // ---- gather: own C_this partials into slots [cc*K + 0], wait workers' slots ----
            // The gather region is K*C_this tiles laid out row-major-by-tile-row: row cc's K
            // partials occupy [cc*K, cc*K + K) (slot cc*K + slice_index for the core with that
            // slice_index). cb_gather has depth K*C_ROWS so full rounds wrap the fifo to base.
            cb_reserve_back(cb_gather, K * C_this);  // blocks until compute popped last round's gather
            const uint32_t gather_base = get_write_ptr(cb_gather);
            for (uint32_t cc = 0; cc < C_this; ++cc) {
                // loopback own partial to slot cc*K + 0 (compacted: only the col-0 faces)
                const uint32_t src = local_src + cc * stat_bytes;
                const uint32_t dst = gather_base + cc * K * stat_bytes;
                noc_async_read(get_noc_addr(master_vx, master_vy, src + G_OFF0), dst + G_OFF0, G_LEN0);
                if (G_LEN1) {
                    noc_async_read(get_noc_addr(master_vx, master_vy, src + G_OFF1), dst + G_OFF1, G_LEN1);
                }
            }
            noc_async_read_barrier();
            cb_pop_front(cb_stat_local, C_this);
            noc_semaphore_wait_min(sem_gather, (r + 1) * (K - 1));  // all worker partials landed
            cb_push_back(cb_gather, K * C_this);                    // -> master compute combine

            // ---- broadcast: wait C_this combined 1/RMS tiles, push to every group member ----
            cb_wait_front(cb_stat_handoff, C_this);
            const uint32_t handoff_src = get_read_ptr(cb_stat_handoff);
            cb_reserve_back(cb_stat_global, C_this);  // blocks until compute popped last round's rstds
            const uint32_t global_dst = get_write_ptr(cb_stat_global);
            noc_semaphore_wait_min(sem_done, (r + 1) * (K - 1));  // all workers ready to receive
            noc_async_read(
                get_noc_addr(master_vx, master_vy, handoff_src), global_dst, bcast_bytes);  // loopback own C_this
            if (n_mcast_seg > 0) {
                // Mcast the C_this finalized 1/RMS tiles to each contiguous virtual segment
                // (1 gap-free rectangle, or 2 runs straddling the DRAM columns). The sender
                // master is auto-excluded from the segment containing it (ndests already
                // accounts for that). Corners in routing order for this NoC: NoC0 walks
                // low->high, NoC1 high->low.
                for (uint32_t s = 0; s < n_mcast_seg; ++s) {
                    const uint32_t sxlo = (s == 0) ? seg0_xlo : seg1_xlo;
                    const uint32_t sylo = (s == 0) ? seg0_ylo : seg1_ylo;
                    const uint32_t sxhi = (s == 0) ? seg0_xhi : seg1_xhi;
                    const uint32_t syhi = (s == 0) ? seg0_yhi : seg1_yhi;
                    const uint32_t snd = (s == 0) ? seg0_nd : seg1_nd;
                    if (snd == 0) {
                        continue;  // segment holds only the master -> nothing to send
                    }
                    const uint64_t mcast_dst = (noc_index == 1)
                                                   ? get_noc_multicast_addr(sxhi, syhi, sxlo, sylo, global_dst)
                                                   : get_noc_multicast_addr(sxlo, sylo, sxhi, syhi, global_dst);
                    noc_async_write_multicast(handoff_src, mcast_dst, bcast_bytes, snd);
                }
            } else {
                for (uint32_t w = 0; w < num_workers; ++w) {
                    const uint32_t wx = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w);
                    const uint32_t wy = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w + 1);
                    noc_async_write(handoff_src, get_noc_addr(wx, wy, global_dst), bcast_bytes);
                }
            }
            noc_async_write_barrier();  // data flushed before the ready signal
            noc_async_read_barrier();
            cb_push_back(cb_stat_global, C_this);  // -> master compute pass 2
            cb_pop_front(cb_stat_handoff, C_this);
            if (n_mcast_seg > 0) {
                // Signal every group member with a mcast semaphore set per segment. Monotone:
                // set the local value to (r+1) (matches the workers' noc_semaphore_wait_min(r+1)),
                // then multicast that value — replacing the K-1 per-worker atomic incs. SEM_DONE
                // back-pressure bounds the master to one round ahead, so cb_stat_global is
                // consumed before the next set overwrites it (no race).
                noc_semaphore_set(sem_bcast, r + 1);
                for (uint32_t s = 0; s < n_mcast_seg; ++s) {
                    const uint32_t sxlo = (s == 0) ? seg0_xlo : seg1_xlo;
                    const uint32_t sylo = (s == 0) ? seg0_ylo : seg1_ylo;
                    const uint32_t sxhi = (s == 0) ? seg0_xhi : seg1_xhi;
                    const uint32_t syhi = (s == 0) ? seg0_yhi : seg1_yhi;
                    const uint32_t snd = (s == 0) ? seg0_nd : seg1_nd;
                    if (snd == 0) {
                        continue;
                    }
                    const uint64_t mcast_sem =
                        (noc_index == 1) ? get_noc_multicast_addr(sxhi, syhi, sxlo, sylo, get_semaphore(SEM_BCAST))
                                         : get_noc_multicast_addr(sxlo, sylo, sxhi, syhi, get_semaphore(SEM_BCAST));
                    noc_semaphore_set_multicast(get_semaphore(SEM_BCAST), mcast_sem, snd);
                }
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
            cb_reserve_back(cb_stat_global, C_this);  // blocks until compute popped last round's rstds
            noc_semaphore_inc(get_noc_addr(master_vx, master_vy, get_semaphore(SEM_DONE)), 1);  // ready to receive
            // master's cb_gather base == this core's cb_gather base (uniform CB offset, empty->base).
            const uint32_t master_gather_base = get_write_ptr(cb_gather);
            for (uint32_t cc = 0; cc < C_this; ++cc) {
                // my partial for row cc -> slot cc*K + slice_index (compacted: col-0 faces only)
                const uint32_t src = local_src + cc * stat_bytes;
                const uint32_t dst = master_gather_base + (cc * K + slice_index) * stat_bytes;
                noc_async_write(src + G_OFF0, get_noc_addr(master_vx, master_vy, dst + G_OFF0), G_LEN0);
                if (G_LEN1) {
                    noc_async_write(src + G_OFF1, get_noc_addr(master_vx, master_vy, dst + G_OFF1), G_LEN1);
                }
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(master_vx, master_vy, get_semaphore(SEM_GATHER)), 1);  // partials landed
            cb_pop_front(cb_stat_local, C_this);
            noc_semaphore_wait_min(sem_bcast, r + 1);  // master broadcast my C_this 1/RMS tiles
            cb_push_back(cb_stat_global, C_this);      // -> worker compute pass 2 (data already at the slots)
        }

        // ---- per-tile-row output drains (OUT_TO_DRAM / IS_RM; C_ROWS=1 on those paths) ----
        for (uint32_t cc = 0; cc < C_this; ++cc) {
            const uint32_t t = base_t + cc;

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
    }
    // Flush the semaphore-inc atomics (SEM_GATHER/SEM_DONE from workers, SEM_BCAST
    // from the master) before the kernel exits — else the target may not observe
    // the final round's signal and the exit "atomics flushed" assert trips.
    noc_async_atomic_barrier();
}
