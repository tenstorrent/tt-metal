// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/hierarchical_gather_r2) -- NOT the op.
//
// ROUND 2 of the cross-core combine TOPOLOGY bake-off.  Round 1 lives next door in
// perf_experiments/hierarchical_gather/; this dir re-measures the same idea against the
// POST-PERF-1 root chain (D16 row-major gather + per-row packer-accumulate fold, D17
// column-scoped fused finalize, D19 finalize-writes-the-handoff), because round 1's own
// note said the absolute win scales down with the per-fold cost and Perf 1 cut that cost.
//
// The combine, and nothing else.  Every core of a group starts with `num_rows` fp32
// partial tiles already resident in its own L1 (a HEIGHT-sharded fp32 input tensor --
// pass A is deliberately not modelled, so the measured delta is attributable to the
// collective alone), and every core must end with the group's finalized stat
// rsqrt(sum_group(partial) * INV_W + eps) in cb_row_final, which is backed on the OUTPUT
// shard -- so the result IS the output tensor.
//
// TWO variants only, because round 2's question is a POLICY question, not a menu:
//
//   0  FLAT   the op's CURRENT approach, carried verbatim (the honest baseline).  Every
//             member ships its partial into its own slot of the single ROOT's gather CB
//             (row-major landing: page = r * GROUP_SIZE + my_slot) and remote-incs the
//             root's arrival semaphore; the root folds GROUP_SIZE partials per row,
//             finalizes all `rows`, and multicasts the stat back.
//
//   1  GRID   ONE unified (K, m) topology whose CORNERS are the whole policy space:
//               (K=1, m=1) == FLAT       (measured, as the generic-path overhead control)
//               (K>1, m=1) == slot TREE  (round 1's `tree_kK`)
//               (K=1, m>1) == ROW SPLIT  (round 1's `rowsplit_wm`)
//               (K>1, m>1) == the genuinely combined point round 1 never measured.
//             The group's GROUP_SIZE slots are cut into K CONTIGUOUS chunks of
//             M = GROUP_SIZE / K, and the row-block is cut into m contiguous row
//             subsets.  Gatherer for (chunk j, row subset w) is slot g(j,w) = j*M + w,
//             so a chunk's gatherers live INSIDE that chunk (hence m <= M) and every
//             gatherer is itself a member of the chunk it gathers -- which fixes its
//             expected arrival count at M - 1 unconditionally, with no self-signal.
//
//             stage 1  every member ships row subset w to g(j(me), w), for every w.
//                      g(j,w) folds M partials per row of subset w.
//             stage 2  (K > 1 only) g(j,w), j > 0, ships its folded partial to g(0,w),
//                      which folds K per row.   [skipped entirely at K == 1]
//             stage 3  (m > 1 only) g(0,w) finalizes subset w and writes the FINISHED
//                      stat tiles straight into the ROOT's cb_row_final at row offset
//                      a_w, then signals.      [skipped entirely at m == 1]
//             mcast    the root (== g(0,0)) broadcasts the assembled block in place.
//
// RING DISCIPLINE.  Every gather ring is sized for ONE worker's share --
// M * RPW pages where RPW = ceil(BLOCK_ROWS / m) -- and the gatherer pushes/pops the
// WHOLE ring every round, so `get_write_ptr` returns the ring BASE at the start of every
// round on every core.  That is what lets a sender compute the landing address LOCALLY
// (the CB is declared identically on every core, so its L1 address is identical
// everywhere) and keeps the host out of CB addresses.  It is also where the L1 win comes
// from: the flat root's ring is GROUP_SIZE * BLOCK_ROWS pages, a row-split gatherer's is
// M * ceil(BLOCK_ROWS/m).
//
// FLOW CONTROL is the multicast, transitively.  A remote sender does not cb_reserve, so
// nothing but the previous round's mcast stops it overwriting a ring the gatherer is still
// folding.  It is safe because the root only sends round `blk`'s stat after every stage-3
// gatherer has finalized, which requires each gatherer's compute to have already drained
// that round's gather ring (and cb_stage2).  The op relies on the identical argument for
// its single root.
//
// NO SELF-SIGNAL, ANYWHERE.  `Semaphore::up(value)` is a NON-ATOMIC local
// read-modify-write, so a local bump on a gatherer would race the members' remote atomic
// incs and silently drop one (the op's writer records this as a hang in one group of
// eight).  Every gatherer writes its own slot synchronously and waits for exactly the
// OTHER contributors.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

namespace {
constexpr uint32_t cb_partials_gathered = 11;  // stage-1 gather ring (fp32 tiles)
constexpr uint32_t cb_stage2 = 12;             // stage-2 gather ring (K > 1)
constexpr uint32_t cb_subroot_out = 13;        // a chunk gatherer's folded partial
constexpr uint32_t cb_row_stat = 14;           // compute-private accumulator
constexpr uint32_t cb_stat_handoff = 15;       // finalized stat, compute -> writer
constexpr uint32_t cb_row_final = 16;          // mcast landing == the output shard
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);     // slot chunks (tree arity)
    constexpr uint32_t MROW = get_compile_time_arg_val(4);  // row-subset gatherers
    constexpr uint32_t SEM1 = get_compile_time_arg_val(5);
    constexpr uint32_t SEM2 = get_compile_time_arg_val(6);
    constexpr uint32_t SEM3 = get_compile_time_arg_val(7);
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(8);
    constexpr uint32_t GMAX = K * MROW;
    constexpr uint32_t RT_MC_BASE = 12 + 2 * GMAX;
    constexpr auto mc = dataflow_kernel_lib::McastArgs<9, RT_MC_BASE>();

    constexpr uint32_t M = GROUP_SIZE / K;
    constexpr uint32_t RPW = (BLOCK_ROWS + MROW - 1) / MROW;
    static_assert(GROUP_SIZE % K == 0, "GRID: K must divide GROUP_SIZE");
    static_assert(MROW <= M, "GRID: a chunk must contain its m gatherers");
    static_assert(GATHER_FACES == 2 || GATHER_FACES == 4, "GATHER_FACES must be 2 or 4");

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t my_slot = get_arg_val<uint32_t>(3);

    // An INACTIVE core: it joined the program only so the stat multicast lands in a
    // cb_row_final this program owns (the op's row-major-packed WIDTH-shard grid does
    // exactly this).  No shard, no work, no ack -- the sender's num_active excludes it.
    if (num_rows == 0) {
        return;
    }

    const uint32_t stat_bytes = get_tile_size(cb_stat_handoff);
    const uint32_t face_bytes = stat_bytes / 4;
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    Noc noc;
    Semaphore<> sem1(SEM1);
    Semaphore<> sem2(SEM2);
    Semaphore<> sem3(SEM3);

    // ---- ONE definition of the partial transfer ------------------------------
    // `src` is indexed by ABSOLUTE tile-row inside this core's resident partial shard;
    // `dst_base` is the destination ring's BASE and the row/slot offset is applied here.
    // `faces == 2` ships only faces 0 and 2 (the pair that can hold a REDUCE_ROW column
    // vector) -- half the bytes, two transactions.  Identical to the op's ship_partial.
    auto ship = [&](uint32_t src,
                    uint64_t dst_base,
                    uint32_t abs_row0,
                    uint32_t rows,
                    uint32_t dst_stride,
                    uint32_t dst_slot,
                    uint32_t faces) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t s_off = (abs_row0 + r) * stat_bytes;
            const uint32_t d_off = (r * dst_stride + dst_slot) * stat_bytes;
            if (faces == 4) {
                noc_async_write(src + s_off, dst_base + d_off, stat_bytes);
            } else {
                noc_async_write(src + s_off, dst_base + d_off, face_bytes);
                noc_async_write(src + s_off + 2 * face_bytes, dst_base + d_off + 2 * face_bytes, face_bytes);
            }
        }
    };

    // Boot: make the faces the gather NEVER writes defined, so no undefined L1 reaches the
    // fold / rsqrt.  Zero EXACTLY the unshipped faces -- zeroing the whole ring races a
    // member's partial that already landed (the op records that as pcc 0.87-0.99 on every
    // combine cell).
    auto zero_unshipped = [&](uint32_t cb) {
        if constexpr (GATHER_FACES == 4) {
            return;
        }
        MaybeDeviceZoneScope("writer_gather_zero");
        DataflowBuffer dfb(cb);
        const uint32_t pages = dfb.get_total_size_bytes() / stat_bytes;
        for (uint32_t p = 0; p < pages; ++p) {
            const uint32_t base = p * stat_bytes;
            noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + face_bytes});
            noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes});
        }
        noc.write_zeros_l1_barrier();
    };

    // The mcast back is identical in both variants: the root places its OWN copy first and
    // broadcasts IN PLACE (src == dst => EXCLUDE-source), which is what makes Mcast1D's
    // per-row rect (excludes the sender) and Mcast2D's rect (contains it) behave the same.

    if constexpr (VARIANT == 0) {
        // ================= FLAT (the op's current approach, verbatim) =============
        if (is_root != 0) {
            zero_unshipped(cb_partials_gathered);
            auto sender = mc.sender(noc);
            uint32_t arrivals = 0;
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_reserve_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
                    ship(
                        in_addr,
                        get_noc_addr(get_write_ptr(cb_partials_gathered)),
                        r0,
                        rows,
                        GROUP_SIZE,
                        my_slot,
                        GATHER_FACES);
                    noc_async_write_barrier();
                }
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arrivals += GROUP_SIZE - 1;
                    sem1.wait_min(arrivals);
                    cb_push_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_send");
                    cb_wait_front(cb_stat_handoff, rows);
                    cb_reserve_back(cb_row_final, rows);
                    const uint32_t dst = get_write_ptr(cb_row_final);
                    noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(dst), rows * stat_bytes);
                    noc_async_write_barrier();
                    if constexpr (mc.active) {
                        sender.send(dst, dst, rows * stat_bytes);
                    }
                    cb_push_back(cb_row_final, rows);
                    cb_pop_front(cb_stat_handoff, rows);
                }
            }
        } else {
            auto receiver = mc.receiver(noc);
            const uint32_t rx = mc.sender_x();
            const uint32_t ry = mc.sender_y();
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    ship(
                        in_addr,
                        get_noc_addr(rx, ry, get_write_ptr(cb_partials_gathered)),
                        r0,
                        rows,
                        GROUP_SIZE,
                        my_slot,
                        GATHER_FACES);
                    noc_async_write_barrier();  // data before signal
                    sem1.up(noc, rx, ry, 1);
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_recv");
                    cb_reserve_back(cb_row_final, rows);
                    receiver.receive();
                    cb_push_back(cb_row_final, rows);
                }
            }
        }
        return;
    }

    // ================= GRID (K slot chunks x m row subsets) =======================
    const uint32_t my_chunk = my_slot / M;  // which slot chunk I contribute to
    const uint32_t my_pos = my_slot % M;    // my slot inside that chunk
    // g(j, w) == j * M + w, so I am the gatherer of (my_chunk, my_pos) iff my_pos < MROW.
    const bool is_gatherer = (my_pos < MROW);
    const uint32_t my_w = my_pos;                       // my row subset, when I gather
    const uint32_t my_j = my_chunk;                     // my chunk column, when I gather
    const bool is_stage2 = is_gatherer && (my_j == 0);  // folds stage 2, finalizes

    // Gatherer virtual coords: table entry gidx = j * MROW + w lives at RT 12 + 2*gidx.
    auto gx = [&](uint32_t gidx) { return get_arg_val<uint32_t>(12 + 2 * gidx); };
    auto gy = [&](uint32_t gidx) { return get_arg_val<uint32_t>(12 + 2 * gidx + 1); };

    if (is_gatherer) {
        zero_unshipped(cb_partials_gathered);
    }

    // Per-round row split, recomputed IDENTICALLY by every core (senders and gatherers):
    // w_eff workers, worker w owns [a_w, a_w + n_w) of the block.
    struct Split {
        uint32_t w_eff, base, extra, my_rows, my_a;
        bool mine;
    };
    auto split_of = [&](uint32_t rows) {
        Split s;
        s.w_eff = (MROW < rows) ? MROW : rows;
        s.base = rows / s.w_eff;
        s.extra = rows % s.w_eff;
        s.mine = is_gatherer && (my_w < s.w_eff);
        s.my_rows = s.mine ? (s.base + (my_w < s.extra ? 1u : 0u)) : 0u;
        s.my_a = s.mine ? (my_w * s.base + (my_w < s.extra ? my_w : s.extra)) : 0u;
        return s;
    };

    uint32_t arr1 = 0, arr2 = 0;

    // Stages 1 and 2 are identical on every core, so they live in one lambda and the
    // root / non-root split below only owns the mcast tail (whose pipe CONSTRUCTOR is the
    // documented handshake init and therefore must not run on the wrong role).
    auto stage12 = [&](uint32_t r0, const Split& s) {
        // ---- stage 1: ship my partial, row subset by row subset -------------------
        {
            MaybeDeviceZoneScope("writer_gather_ship");
            if (s.mine) {
                cb_reserve_back(cb_partials_gathered, M * RPW);
            }
            uint32_t a = 0;
            for (uint32_t w = 0; w < s.w_eff; ++w) {
                const uint32_t nw = s.base + (w < s.extra ? 1u : 0u);
                const uint32_t gidx = my_chunk * MROW + w;
                const uint64_t dst = (is_gatherer && w == my_w)
                                         ? get_noc_addr(get_write_ptr(cb_partials_gathered))
                                         : get_noc_addr(gx(gidx), gy(gidx), get_write_ptr(cb_partials_gathered));
                ship(in_addr, dst, r0 + a, nw, M, my_pos, GATHER_FACES);
                a += nw;
            }
            noc_async_write_barrier();  // data before ALL signals
            for (uint32_t w = 0; w < s.w_eff; ++w) {
                if (is_gatherer && w == my_w) {
                    continue;  // NO self-signal (non-atomic local RMW)
                }
                const uint32_t gidx = my_chunk * MROW + w;
                sem1.up(noc, gx(gidx), gy(gidx), 1);
            }
        }
        if (s.mine) {
            MaybeDeviceZoneScope("writer_gather_wait");
            arr1 += M - 1;
            sem1.wait_min(arr1);
            cb_push_back(cb_partials_gathered, M * RPW);
        }

        // ---- stage 2: chunk gatherers fold-forward to g(0, w) --------------------
        if constexpr (K > 1) {
            if (s.mine) {
                if (my_j != 0) {
                    MaybeDeviceZoneScope("writer_stage2_ship");
                    cb_wait_front(cb_subroot_out, s.my_rows);
                    const uint32_t g0 = my_w;  // g(0, my_w)
                    // WHOLE tiles: cb_subroot_out's faces 1/3 are the stage-1 ring's boot
                    // zeros folded through DEST, so stage 2 needs no second zeroing pass
                    // and is ONE transaction per row.  K is small next to GROUP_SIZE.
                    ship(
                        get_read_ptr(cb_subroot_out),
                        get_noc_addr(gx(g0), gy(g0), get_write_ptr(cb_stage2)),
                        0,
                        s.my_rows,
                        K,
                        my_j,
                        4);
                    noc_async_write_barrier();  // data before signal
                    sem2.up(noc, gx(g0), gy(g0), 1);
                    cb_pop_front(cb_subroot_out, s.my_rows);
                } else {
                    {
                        MaybeDeviceZoneScope("writer_stage2_ship");
                        cb_wait_front(cb_subroot_out, s.my_rows);
                        cb_reserve_back(cb_stage2, K * RPW);
                        ship(
                            get_read_ptr(cb_subroot_out),
                            get_noc_addr(get_write_ptr(cb_stage2)),
                            0,
                            s.my_rows,
                            K,
                            0,
                            4);
                        noc_async_write_barrier();
                        cb_pop_front(cb_subroot_out, s.my_rows);
                    }
                    {
                        MaybeDeviceZoneScope("writer_stage2_wait");
                        arr2 += K - 1;
                        sem2.wait_min(arr2);
                        cb_push_back(cb_stage2, K * RPW);
                    }
                }
            }
        }
    };

    if (is_root != 0) {
        auto sender = mc.sender(noc);
        uint32_t arr3 = 0;
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t r0 = blk * BLOCK_ROWS;
            const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
            const Split s = split_of(rows);
            stage12(r0, s);
            MaybeDeviceZoneScope("writer_mcast_send");
            cb_wait_front(cb_stat_handoff, s.my_rows);
            cb_reserve_back(cb_row_final, rows);
            const uint32_t dst = get_write_ptr(cb_row_final);
            // My finished rows land at row 0 of the block (a_0 == 0); every other stage-3
            // gatherer writes its own rows straight into this same buffer, so the mcast
            // source is assembled IN PLACE.
            noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(dst), s.my_rows * stat_bytes);
            noc_async_write_barrier();
            if constexpr (MROW > 1) {
                MaybeDeviceZoneScope("writer_stage3_wait");
                arr3 += s.w_eff - 1;
                sem3.wait_min(arr3);
            }
            if constexpr (mc.active) {
                sender.send(dst, dst, rows * stat_bytes);
            }
            cb_push_back(cb_row_final, rows);
            cb_pop_front(cb_stat_handoff, s.my_rows);
        }
    } else {
        auto receiver = mc.receiver(noc);
        const uint32_t rx = mc.sender_x();  // the group root == g(0,0)
        const uint32_t ry = mc.sender_y();
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t r0 = blk * BLOCK_ROWS;
            const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
            const Split s = split_of(rows);
            stage12(r0, s);
            if (is_stage2 && s.mine) {
                MaybeDeviceZoneScope("writer_stage3_ship");
                cb_wait_front(cb_stat_handoff, s.my_rows);
                // Every core pushes `rows` to cb_row_final per round, so this core's
                // get_write_ptr(cb_row_final) IS the root's.
                const uint32_t dst = get_write_ptr(cb_row_final) + s.my_a * stat_bytes;
                noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(rx, ry, dst), s.my_rows * stat_bytes);
                noc_async_write_barrier();  // data before signal
                sem3.up(noc, rx, ry, 1);
                cb_pop_front(cb_stat_handoff, s.my_rows);
            }
            {
                MaybeDeviceZoneScope("writer_mcast_recv");
                cb_reserve_back(cb_row_final, rows);
                receiver.receive();
                cb_push_back(cb_row_final, rows);
            }
        }
    }
}
