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
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_shard_out = 9;  // RM output: zero-copy alias of the resident RM W-slice (stick pages)
constexpr uint32_t cb_gather = 5;
constexpr uint32_t cb_stat_handoff = 6;
constexpr uint32_t cb_stat_global = 7;
constexpr uint32_t cb_rowpartial = 10;  // R6c two-stage: row-leader stage-1 fold output (compute -> writer)
constexpr uint32_t cb_gather2 = 11;     // R6c two-stage: root stage-2 fan-in (ny row-partials, fixed base)
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
    // TWO_STAGE (Refinement 6c): hierarchical gather for a 2D group (NX x NY, both > 1). Replaces
    // the flat K-1 -> master gather with stage 1 (each grid row's NX cores -> the row's x0
    // row-leader, folded to a row-partial) then stage 2 (the NY row-leaders -> the root, folded +
    // finalized). Fan-in drops from K-1 to (NX-1)+(NY-1) and the fold is distributed off the single
    // master. Host-gated to single-round (C=1, HT_LOCAL=1) clean rectangles; the broadcast leg
    // (root -> all members) reuses the R6/R6a segmented mcast unchanged. TWO_STAGE=0 -> the flat
    // path below is byte-identical.
    constexpr bool TWO_STAGE = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t NX = get_compile_time_arg_val(19);           // cores per grid row (stage-1 fan-in)
    constexpr uint32_t NY = get_compile_time_arg_val(20);           // grid rows (stage-2 fan-in)
    constexpr uint32_t SEM_GATHER2 = get_compile_time_arg_val(21);  // stage-2 gather counter id
    // TWO_PHASE_FOLD (Refinement 6e): distribute the master's serial K-partial fold across
    // NUM_FOLDERS = min(C_ROWS, K) folder cores by tile-index. Every core pushes each tile-row's
    // partial to that row's folder (owner = row % NUM_FOLDERS); each folder gathers K partials for
    // its owned rows, compute folds them (+eps, rsqrt), the folder scatters the finalized 1/RMS
    // tiles to the root's cb_stat_global, and the root assembles all C and mcasts them back
    // (segmented, R6/R6a). SEM reuse: SEM_GATHER (cores->folder, folder waits (r+1)*K), SEM_GATHER2
    // (folders->root, root waits (r+1)*NUM_FOLDERS), SEM_BCAST (root->non-root, set r+1). Both
    // cb_stat_global back-pressures are FREE: the round is fully synchronous and the gather barrier
    // includes every core (incl root), whose round-r push is gated behind its own pass-2 r-1 pop.
    // Host-gated to the pure tiled BLOCK path with C>1 multi-round + a valid mcast segment; the flat
    // path below is byte-identical when TWO_PHASE_FOLD=0.
    constexpr bool TWO_PHASE_FOLD = get_compile_time_arg_val(22) != 0;
    constexpr uint32_t NUM_FOLDERS = get_compile_time_arg_val(23);

    constexpr auto out_args = TensorAccessorArgs<24>();

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
    // R6c two-stage per-core fields (0 when TWO_STAGE off): role 2=root, 1=row-leader, 0=worker;
    // xrel = stage-1 gather slot (column within row); yrel = stage-2 gather slot (row index);
    // (rl_vx,rl_vy) = this core's row-leader VIRTUAL coords (a row-leader/root gets its own coords,
    // the self-loopback source — mirroring how the flat master loopbacks via master_vx/vy).
    const uint32_t role = get_arg_val<uint32_t>(22);
    const uint32_t xrel = get_arg_val<uint32_t>(23);
    const uint32_t yrel = get_arg_val<uint32_t>(24);
    const uint32_t rl_vx = get_arg_val<uint32_t>(25);
    const uint32_t rl_vy = get_arg_val<uint32_t>(26);
    // master only: worker virtual coords follow as [vx, vy] * num_workers (used by the
    // all-unicast fallback; unused on the mcast path but always emitted by the host).
    constexpr uint32_t WORKER_COORDS_BASE = 27;

    volatile tt_l1_ptr uint32_t* sem_gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));
    volatile tt_l1_ptr uint32_t* sem_bcast = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BCAST));
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE));
    volatile tt_l1_ptr uint32_t* sem_gather2 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER2));

    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page);

    // ======================= Refinement 6e: two-phase distributed fold =======================
    // Every core pushes each batched tile-row's partial to that row's FOLDER (owner = row %
    // NUM_FOLDERS) instead of all to one master; each folder gathers K partials for its owned
    // rows, compute folds them (+eps, rsqrt), and the folder scatters the finalized 1/RMS tiles
    // to the root's cb_stat_global. The root assembles all C_this and mcasts them back (segmented,
    // R6/R6a). Fully synchronous per round; host gates Ht_local % C_ROWS == 0 so every folder owns
    // owned_max rows every round (monotone semaphore counts stay uniform). The output shard is a
    // zero-copy CB on this path (no drain). Both cb_stat_global back-pressures are free: the round
    // barrier + the gather including every core (whose round-r push is gated behind its own pass-2
    // r-1 pop) guarantee the buffer is drained before any round-r remote write lands.
    if constexpr (TWO_PHASE_FOLD) {
        const uint32_t is_folder = get_arg_val<uint32_t>(27);
        const uint32_t owned_max = get_arg_val<uint32_t>(28);
        constexpr uint32_t FOLDER_COORDS_BASE = 29;
        const uint32_t num_rounds = (HT_LOCAL + C_ROWS - 1) / C_ROWS;
        for (uint32_t r = 0; r < num_rounds; ++r) {
            MaybeDeviceZoneScope("xc_wr_round");
            const uint32_t base_t = r * C_ROWS;
            uint32_t C_this = HT_LOCAL - base_t;
            if (C_this > C_ROWS) {
                C_this = C_ROWS;
            }
            const uint32_t bcast_bytes = C_this * stat_bytes;

            cb_wait_front(cb_stat_local, C_this);  // compute produced C_this local partials
            const uint32_t local_src = get_read_ptr(cb_stat_local);

            // ---- gather-push: send row cc's partial to folder (cc % NUM_FOLDERS), slot l*K+si ----
            for (uint32_t f = 0; f < NUM_FOLDERS; ++f) {
                const uint32_t fvx = get_arg_val<uint32_t>(FOLDER_COORDS_BASE + 2 * f);
                const uint32_t fvy = get_arg_val<uint32_t>(FOLDER_COORDS_BASE + 2 * f + 1);
                const uint32_t fbase = get_write_ptr(cb_gather);  // uniform CB base (empty -> base)
                uint32_t l = 0;
                for (uint32_t cc = f; cc < C_this; cc += NUM_FOLDERS) {
                    const uint32_t src = local_src + cc * stat_bytes;
                    const uint32_t dst = fbase + (l * K + slice_index) * stat_bytes;
                    noc_async_write(src + G_OFF0, get_noc_addr(fvx, fvy, dst + G_OFF0), G_LEN0);
                    if (G_LEN1) {
                        noc_async_write(src + G_OFF1, get_noc_addr(fvx, fvy, dst + G_OFF1), G_LEN1);
                    }
                    ++l;
                }
            }
            noc_async_write_barrier();
            for (uint32_t f = 0; f < NUM_FOLDERS; ++f) {
                const uint32_t fvx = get_arg_val<uint32_t>(FOLDER_COORDS_BASE + 2 * f);
                const uint32_t fvy = get_arg_val<uint32_t>(FOLDER_COORDS_BASE + 2 * f + 1);
                noc_semaphore_inc(get_noc_addr(fvx, fvy, get_semaphore(SEM_GATHER)), 1);
            }
            cb_pop_front(cb_stat_local, C_this);

            // ---- folder: gather K partials/owned-row -> compute fold -> scatter rstds to root ----
            if (is_folder) {
                cb_reserve_back(cb_gather, owned_max * K);        // blocks until compute popped prev
                noc_semaphore_wait_min(sem_gather, (r + 1) * K);  // all K cores pushed my owned rows
                cb_push_back(cb_gather, owned_max * K);           // -> compute do_fold_owned

                cb_wait_front(cb_rowpartial, owned_max);  // compute finalized my owned rstds
                const uint32_t rp_src = get_read_ptr(cb_rowpartial);
                const uint32_t root_global_base = get_write_ptr(cb_stat_global);  // uniform base
                uint32_t l = 0;
                for (uint32_t cc = slice_index; cc < C_this; cc += NUM_FOLDERS) {
                    // scatter finalized 1/RMS for owned row cc -> root's cb_stat_global[cc] (full tile)
                    const uint32_t src = rp_src + l * stat_bytes;
                    const uint32_t dst = root_global_base + cc * stat_bytes;
                    noc_async_write(src, get_noc_addr(master_vx, master_vy, dst), stat_bytes);
                    ++l;
                }
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(master_vx, master_vy, get_semaphore(SEM_GATHER2)), 1);
                cb_pop_front(cb_rowpartial, owned_max);
            }

            // ---- root: assemble all C_this rstds + broadcast (segmented mcast) ----
            if (is_master) {
                cb_reserve_back(cb_stat_global, C_this);  // bookkeeping; buffer already free (round barrier)
                noc_semaphore_wait_min(sem_gather2, (r + 1) * NUM_FOLDERS);  // all folders scattered
                const uint32_t global_src = get_write_ptr(cb_stat_global);   // assembled C_this rstds at base
                for (uint32_t s = 0; s < n_mcast_seg; ++s) {
                    const uint32_t sxlo = (s == 0) ? seg0_xlo : seg1_xlo;
                    const uint32_t sylo = (s == 0) ? seg0_ylo : seg1_ylo;
                    const uint32_t sxhi = (s == 0) ? seg0_xhi : seg1_xhi;
                    const uint32_t syhi = (s == 0) ? seg0_yhi : seg1_yhi;
                    const uint32_t snd = (s == 0) ? seg0_nd : seg1_nd;
                    if (snd == 0) {
                        continue;
                    }
                    const uint64_t mcast_dst = (noc_index == 1)
                                                   ? get_noc_multicast_addr(sxhi, syhi, sxlo, sylo, global_src)
                                                   : get_noc_multicast_addr(sxlo, sylo, sxhi, syhi, global_src);
                    noc_async_write_multicast(global_src, mcast_dst, bcast_bytes, snd);
                }
                noc_async_write_barrier();
                cb_push_back(cb_stat_global, C_this);  // -> root compute pass 2
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
                noc_async_write_barrier();
            } else {
                cb_reserve_back(cb_stat_global, C_this);   // bookkeeping; buffer already free
                noc_semaphore_wait_min(sem_bcast, r + 1);  // root broadcast landed
                cb_push_back(cb_stat_global, C_this);      // -> compute pass 2
            }
        }
        noc_async_atomic_barrier();  // flush the semaphore-inc atomics before exit
        return;
    }
    // ===================== end Refinement 6e two-phase distributed fold =====================

    // ======================= R6c two-stage (hierarchical) gather =======================
    // Single round (host-gated to C=1, HT_LOCAL=1). The output is a zero-copy sharded CB
    // (OUT_TO_DRAM=0, IS_RM=0 on this path), so there is no output drain — only the stat
    // transport. Compaction (G_OFF*/G_LEN*) applies to BOTH gather legs; the broadcast leg
    // moves the full stat tile (reuses the R6/R6a segmented mcast unchanged).
    if constexpr (TWO_STAGE) {
        MaybeDeviceZoneScope("xc_wr_round");
        cb_wait_front(cb_stat_local, 1);  // compute produced this core's local partial
        const uint32_t local_src = get_read_ptr(cb_stat_local);

        if (role == 0) {
            // ---- worker: send partial to its row-leader's cb_gather[xrel], await broadcast ----
            cb_reserve_back(cb_stat_global, 1);                        // position the receive slot (fixed base)
            const uint32_t rl_gather_base = get_write_ptr(cb_gather);  // uniform CB base across cores
            const uint32_t dst = rl_gather_base + xrel * stat_bytes;
            noc_async_write(local_src + G_OFF0, get_noc_addr(rl_vx, rl_vy, dst + G_OFF0), G_LEN0);
            if (G_LEN1) {
                noc_async_write(local_src + G_OFF1, get_noc_addr(rl_vx, rl_vy, dst + G_OFF1), G_LEN1);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rl_vx, rl_vy, get_semaphore(SEM_GATHER)), 1);
            cb_pop_front(cb_stat_local, 1);
            noc_semaphore_wait_min(sem_bcast, 1);  // root broadcast the finalized 1/RMS
            cb_push_back(cb_stat_global, 1);
        } else {
            // ---- row-leader (role 1) or root (role 2): stage-1 gather this grid row ----
            cb_reserve_back(cb_gather, NX);  // fixed-base fan-in (NX partials, this row)
            const uint32_t gbase = get_write_ptr(cb_gather);
            // loopback own partial into slot 0 (rl_vx/rl_vy == this core's own virtual coords)
            noc_async_read(get_noc_addr(rl_vx, rl_vy, local_src + G_OFF0), gbase + G_OFF0, G_LEN0);
            if (G_LEN1) {
                noc_async_read(get_noc_addr(rl_vx, rl_vy, local_src + G_OFF1), gbase + G_OFF1, G_LEN1);
            }
            noc_async_read_barrier();
            cb_pop_front(cb_stat_local, 1);
            noc_semaphore_wait_min(sem_gather, NX - 1);  // this row's workers landed
            cb_push_back(cb_gather, NX);                 // -> compute stage-1 fold -> cb_rowpartial

            cb_wait_front(cb_rowpartial, 1);  // compute folded the NX partials -> row-partial
            const uint32_t rp_src = get_read_ptr(cb_rowpartial);

            if (role == 1) {
                // ---- row-leader: send row-partial to root's cb_gather2[yrel], await broadcast ----
                cb_reserve_back(cb_stat_global, 1);
                const uint32_t g2base = get_write_ptr(cb_gather2);  // uniform base across cores
                const uint32_t dst2 = g2base + yrel * stat_bytes;
                noc_async_write(rp_src + G_OFF0, get_noc_addr(master_vx, master_vy, dst2 + G_OFF0), G_LEN0);
                if (G_LEN1) {
                    noc_async_write(rp_src + G_OFF1, get_noc_addr(master_vx, master_vy, dst2 + G_OFF1), G_LEN1);
                }
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(master_vx, master_vy, get_semaphore(SEM_GATHER2)), 1);
                cb_pop_front(cb_rowpartial, 1);
                noc_semaphore_wait_min(sem_bcast, 1);
                cb_push_back(cb_stat_global, 1);
            } else {
                // ---- root: stage-2 gather the NY row-partials, fold+finalize, broadcast ----
                cb_reserve_back(cb_gather2, NY);
                const uint32_t g2base = get_write_ptr(cb_gather2);
                // loopback own row-partial into slot 0 (master_vx/vy == root's own coords)
                noc_async_read(get_noc_addr(master_vx, master_vy, rp_src + G_OFF0), g2base + G_OFF0, G_LEN0);
                if (G_LEN1) {
                    noc_async_read(get_noc_addr(master_vx, master_vy, rp_src + G_OFF1), g2base + G_OFF1, G_LEN1);
                }
                noc_async_read_barrier();
                cb_pop_front(cb_rowpartial, 1);
                noc_semaphore_wait_min(sem_gather2, NY - 1);  // row-leaders landed
                cb_push_back(cb_gather2, NY);  // -> compute stage-2 fold (+eps, rsqrt) -> cb_stat_handoff

                // ---- broadcast: root sends the finalized 1/RMS to every group member ----
                cb_wait_front(cb_stat_handoff, 1);
                const uint32_t handoff_src = get_read_ptr(cb_stat_handoff);
                cb_reserve_back(cb_stat_global, 1);
                const uint32_t global_dst = get_write_ptr(cb_stat_global);
                noc_async_read(get_noc_addr(master_vx, master_vy, handoff_src), global_dst, stat_bytes);  // own copy
                if (n_mcast_seg > 0) {
                    for (uint32_t s = 0; s < n_mcast_seg; ++s) {
                        const uint32_t sxlo = (s == 0) ? seg0_xlo : seg1_xlo;
                        const uint32_t sylo = (s == 0) ? seg0_ylo : seg1_ylo;
                        const uint32_t sxhi = (s == 0) ? seg0_xhi : seg1_xhi;
                        const uint32_t syhi = (s == 0) ? seg0_yhi : seg1_yhi;
                        const uint32_t snd = (s == 0) ? seg0_nd : seg1_nd;
                        if (snd == 0) {
                            continue;
                        }
                        const uint64_t mcast_dst = (noc_index == 1)
                                                       ? get_noc_multicast_addr(sxhi, syhi, sxlo, sylo, global_dst)
                                                       : get_noc_multicast_addr(sxlo, sylo, sxhi, syhi, global_dst);
                        noc_async_write_multicast(handoff_src, mcast_dst, stat_bytes, snd);
                    }
                } else {
                    for (uint32_t w = 0; w < num_workers; ++w) {
                        const uint32_t wx = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w);
                        const uint32_t wy = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w + 1);
                        noc_async_write(handoff_src, get_noc_addr(wx, wy, global_dst), stat_bytes);
                    }
                }
                noc_async_write_barrier();
                noc_async_read_barrier();
                cb_push_back(cb_stat_global, 1);  // -> root compute pass 2
                cb_pop_front(cb_stat_handoff, 1);
                // signal every member the broadcast landed (monotone set = 1, single round)
                if (n_mcast_seg > 0) {
                    noc_semaphore_set(sem_bcast, 1);
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
                    noc_async_write_barrier();
                } else {
                    for (uint32_t w = 0; w < num_workers; ++w) {
                        const uint32_t wx = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w);
                        const uint32_t wy = get_arg_val<uint32_t>(WORKER_COORDS_BASE + 2 * w + 1);
                        noc_semaphore_inc(get_noc_addr(wx, wy, get_semaphore(SEM_BCAST)), 1);
                    }
                }
            }
        }
        noc_async_atomic_barrier();  // flush semaphore-inc atomics before exit
        return;
    }
    // ===================== end R6c two-stage gather =====================

    const uint32_t num_rounds = (HT_LOCAL + C_ROWS - 1) / C_ROWS;
    for (uint32_t r = 0; r < num_rounds; ++r) {
        MaybeDeviceZoneScope("xc_wr_round");
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
