// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/hierarchical_gather) -- NOT the op.
//
// The rms_norm cross-core width COMBINE, and nothing else.  Every core of a group
// starts with `num_rows` fp32 partial tiles already resident in its own L1 (the
// input shard: pass A is deliberately not modelled) and every core must end with
// the group's finalized stat = rsqrt(sum_group(partial) * INV_W + eps) in
// cb_row_final (which is backed on the output shard, so the result IS the tensor).
//
// Three topologies, selected by VARIANT:
//
//   0  FLAT      the op's current approach.  Every member ships its partial into
//                its own slot of the ROOT's gather CB and remote-incs the root's
//                arrival semaphore; the root sums GROUP_SIZE partials per row,
//                finalizes, and multicasts the stat back.  The root's serial add
//                chain is GROUP_SIZE long.
//
//   1  TREE      two-stage hierarchy.  The group's GROUP_SIZE slots are cut into K
//                CONTIGUOUS chunks of M = GROUP_SIZE / K; the first slot of each
//                chunk is a SUB-ROOT.  Stage 1: each member ships to its sub-root,
//                which folds M partials per row.  Stage 2: the K sub-roots ship
//                their folded partials to the group root, which folds K per row,
//                finalizes once, and multicasts as before.  Serial chain
//                M + K instead of GROUP_SIZE.
//                Because slots are assigned row-major over the group's core
//                rectangle, a contiguous slot chunk is a grid-row prefix -- so
//                K == (rows of the group rectangle) IS `two_stage_grid_reduce`
//                (reduce along grid-x, then along grid-y).  On a 1-D group
//                (gy == 1) the grid-axis variant has only one axis and collapses
//                to FLAT; K is then a pure slot-chunk hierarchy.
//
//   2  ROWSPLIT  the same flat fan-in, but the ROW axis of the block is split over
//                W_MAX workers (slots 0..w-1, w = min(W_MAX, rows)).  Worker i
//                gathers all GROUP_SIZE partials for ITS rows only, folds AND
//                finalizes them, and writes the finished stat tiles straight into
//                the root's cb_row_final at its row offset.  The root then
//                multicasts the assembled block IN PLACE.  Per-row fan-in is
//                unchanged (GROUP_SIZE) but the root's serial row count drops from
//                `rows` to `rows / w`, which is the other half of the same
//                critical path (root_sum + root_finalize + stat_handoff).
//
// RING DISCIPLINE (all three).  A gather CB is sized FANIN * BLOCK_ROWS pages and
// the gatherer pushes/pops the WHOLE ring every round, so `get_write_ptr` returns
// the ring BASE at the start of every round on every core -- which is what lets a
// sender compute the landing address LOCALLY (the CB is declared on every core, so
// its L1 address is identical everywhere) and keeps the host out of CB addresses.
// The op relies on the same identity but pushes only GROUP_SIZE * rows, which
// happens to wrap only because its ragged block is always the last one; pushing
// the whole ring makes the identity unconditional and is what the uneven ROWSPLIT
// row shares need.  The gatherer's compute pops the unused tail.
//
// NO SELF-SIGNAL, ANYWHERE.  `Semaphore::up(value)` is a NON-ATOMIC local
// read-modify-write, so a local bump on a gatherer would race the members' remote
// atomic incs and silently drop one (the op's kernel records this as a hang in one
// group of eight).  Every gatherer writes its own slot synchronously and waits for
// exactly the OTHER contributors.

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
constexpr uint32_t cb_stage2 = 12;             // TREE stage-2 gather ring
constexpr uint32_t cb_subroot_out = 13;        // TREE: sub-root's folded partial
constexpr uint32_t cb_row_stat = 14;           // compute-private accumulator
constexpr uint32_t cb_stat_handoff = 15;       // finalized stat, compute -> writer
constexpr uint32_t cb_row_final = 16;          // mcast landing == the output shard
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);  // TREE branching factor
    constexpr uint32_t SEM1 = get_compile_time_arg_val(4);
    constexpr uint32_t SEM2 = get_compile_time_arg_val(5);
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(6);
    constexpr uint32_t W_MAX = get_compile_time_arg_val(7);  // ROWSPLIT workers
    constexpr uint32_t RT_MC_BASE = 12 + 2 * W_MAX;
    constexpr auto mc = dataflow_kernel_lib::McastArgs<8, RT_MC_BASE>();

    constexpr uint32_t M = (VARIANT == 1) ? (GROUP_SIZE / K) : GROUP_SIZE;
    static_assert(VARIANT != 1 || (GROUP_SIZE % K) == 0, "TREE: K must divide GROUP_SIZE");
    static_assert(GATHER_FACES == 2 || GATHER_FACES == 4, "GATHER_FACES must be 2 or 4");

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t my_slot = get_arg_val<uint32_t>(3);
    const uint32_t my_pos = get_arg_val<uint32_t>(4);       // slot within my sub-group
    const uint32_t my_subgroup = get_arg_val<uint32_t>(5);  // sub-group index j
    const uint32_t is_subroot = get_arg_val<uint32_t>(6);
    const uint32_t subroot_x = get_arg_val<uint32_t>(7);
    const uint32_t subroot_y = get_arg_val<uint32_t>(8);

    // An INACTIVE core: it joined the program only so the stat multicast lands in
    // a cb_row_final this program owns (the op's row-major-packed WIDTH-shard grid
    // does exactly this).  No shard, no work, no ack -- the sender's num_active
    // excludes it.
    if (num_rows == 0) {
        return;
    }

    const uint32_t stat_bytes = get_tile_size(cb_stat_handoff);
    const uint32_t face_bytes = stat_bytes / 4;
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    Noc noc;
    Semaphore<> sem1(SEM1);
    Semaphore<> sem2(SEM2);

    // ---- ONE definition of the partial transfer ------------------------------
    // `src` is indexed by ABSOLUTE tile-row inside this core's resident partial
    // shard; `dst_base` is the destination ring's BASE and the row/slot offset is
    // applied here.  `faces == 2` ships only faces 0 and 2 (the pair that can hold
    // a REDUCE_ROW column vector) -- half the bytes, two transactions.
    auto ship = [&](uint32_t src,
                    uint64_t dst_base,
                    uint32_t abs_row0,
                    uint32_t rows,
                    uint32_t dst_stride,
                    uint32_t dst_slot,
                    uint32_t dst_row0,
                    uint32_t faces) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t s_off = (abs_row0 + r) * stat_bytes;
            const uint32_t d_off = ((dst_row0 + r) * dst_stride + dst_slot) * stat_bytes;
            if (faces == 4) {
                noc_async_write(src + s_off, dst_base + d_off, stat_bytes);
            } else {
                noc_async_write(src + s_off, dst_base + d_off, face_bytes);
                noc_async_write(src + s_off + 2 * face_bytes, dst_base + d_off + 2 * face_bytes, face_bytes);
            }
        }
    };

    // Boot: make the faces the gather NEVER writes defined, so no undefined L1
    // reaches the fold / rsqrt.  Zero EXACTLY the unshipped faces -- zeroing the
    // whole ring races a member's partial that already landed (the op records that
    // as pcc 0.87-0.99 on every combine cell).
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

    // The mcast back is identical in all three variants: the root places its OWN
    // copy first and broadcasts IN PLACE (src == dst => EXCLUDE-source), which is
    // what makes Mcast1D's per-row rect (excludes the sender) and Mcast2D's rect
    // (contains it) behave identically.

    if constexpr (VARIANT == 0) {
        // =================== FLAT ROOT (the op's current approach) ============
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
                        0,
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
                        0,
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
    } else if constexpr (VARIANT == 1) {
        // =================== TREE (two-stage hierarchy) =======================
        if (is_subroot != 0) {
            zero_unshipped(cb_partials_gathered);
        }
        // stage-2 ships WHOLE tiles: cb_subroot_out's faces 1/3 are the stage-1
        // ring's boot zeros folded through DEST, so a 4-face stage-2 transfer needs
        // no second zeroing pass and is ONE transaction per row.  K is small, so
        // the extra bytes are negligible next to the GROUP_SIZE-wide stage 1.
        if (is_root != 0) {
            auto sender = mc.sender(noc);
            uint32_t arr1 = 0, arr2 = 0;
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                // -- stage 1: gather my sub-group's M partials (I am sub-root 0) --
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_reserve_back(cb_partials_gathered, M * BLOCK_ROWS);
                    ship(
                        in_addr,
                        get_noc_addr(get_write_ptr(cb_partials_gathered)),
                        r0,
                        rows,
                        M,
                        my_pos,
                        0,
                        GATHER_FACES);
                    noc_async_write_barrier();
                }
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arr1 += M - 1;
                    sem1.wait_min(arr1);
                    cb_push_back(cb_partials_gathered, M * BLOCK_ROWS);
                }
                // -- stage 2: my folded partial into my own slot 0, then wait ----
                {
                    MaybeDeviceZoneScope("writer_stage2_ship");
                    cb_wait_front(cb_subroot_out, rows);
                    cb_reserve_back(cb_stage2, K * BLOCK_ROWS);
                    ship(get_read_ptr(cb_subroot_out), get_noc_addr(get_write_ptr(cb_stage2)), 0, rows, K, 0, 0, 4);
                    noc_async_write_barrier();
                    cb_pop_front(cb_subroot_out, rows);
                }
                {
                    MaybeDeviceZoneScope("writer_stage2_wait");
                    arr2 += K - 1;
                    sem2.wait_min(arr2);
                    cb_push_back(cb_stage2, K * BLOCK_ROWS);
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
        } else if (is_subroot != 0) {
            // A non-root SUB-ROOT: gathers M, folds, forwards to the root, and is
            // itself an mcast receiver.
            auto receiver = mc.receiver(noc);
            const uint32_t rx = mc.sender_x();  // the group root
            const uint32_t ry = mc.sender_y();
            uint32_t arr1 = 0;
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_reserve_back(cb_partials_gathered, M * BLOCK_ROWS);
                    ship(
                        in_addr,
                        get_noc_addr(get_write_ptr(cb_partials_gathered)),
                        r0,
                        rows,
                        M,
                        my_pos,
                        0,
                        GATHER_FACES);
                    noc_async_write_barrier();
                }
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arr1 += M - 1;
                    sem1.wait_min(arr1);
                    cb_push_back(cb_partials_gathered, M * BLOCK_ROWS);
                }
                {
                    MaybeDeviceZoneScope("writer_stage2_ship");
                    cb_wait_front(cb_subroot_out, rows);
                    ship(
                        get_read_ptr(cb_subroot_out),
                        get_noc_addr(rx, ry, get_write_ptr(cb_stage2)),
                        0,
                        rows,
                        K,
                        my_subgroup,
                        0,
                        4);
                    noc_async_write_barrier();  // data before signal
                    sem2.up(noc, rx, ry, 1);
                    cb_pop_front(cb_subroot_out, rows);
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_recv");
                    cb_reserve_back(cb_row_final, rows);
                    receiver.receive();
                    cb_push_back(cb_row_final, rows);
                }
            }
        } else {
            // A plain member: ships to its SUB-ROOT (not the root) and receives.
            auto receiver = mc.receiver(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    ship(
                        in_addr,
                        get_noc_addr(subroot_x, subroot_y, get_write_ptr(cb_partials_gathered)),
                        r0,
                        rows,
                        M,
                        my_pos,
                        0,
                        GATHER_FACES);
                    noc_async_write_barrier();
                    sem1.up(noc, subroot_x, subroot_y, 1);
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_recv");
                    cb_reserve_back(cb_row_final, rows);
                    receiver.receive();
                    cb_push_back(cb_row_final, rows);
                }
            }
        }
    } else {
        // =================== ROWSPLIT (row-parallel workers) =================
        // w = min(W_MAX, rows) workers per round; worker i == slot i owns block-local
        // rows [a_i, b_i).  Every core ships each worker exactly that worker's rows,
        // takes ONE barrier, then incs each remote worker's arrival semaphore once.
        const uint32_t is_worker_slot = (my_slot < W_MAX) ? 1u : 0u;
        uint32_t arr1 = 0;
        uint32_t arr2 = 0;
        if (is_worker_slot != 0) {
            zero_unshipped(cb_partials_gathered);
        }

        // Worker virtual coords live at RT 12 + 2*i.
        auto wx = [&](uint32_t i) { return get_arg_val<uint32_t>(12 + 2 * i); };
        auto wy = [&](uint32_t i) { return get_arg_val<uint32_t>(12 + 2 * i + 1); };

        if (is_root != 0) {
            auto sender = mc.sender(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                const uint32_t w = (W_MAX < rows) ? W_MAX : rows;
                const uint32_t base = rows / w, extra = rows % w;
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    // my own rows (worker 0 == root: a_0 == 0) go into my ring
                    const uint32_t my_rows = base + (0 < extra ? 1u : 0u);
                    cb_reserve_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
                    ship(
                        in_addr,
                        get_noc_addr(get_write_ptr(cb_partials_gathered)),
                        r0,
                        my_rows,
                        GROUP_SIZE,
                        my_slot,
                        0,
                        GATHER_FACES);
                    // every OTHER worker gets only its own row range
                    uint32_t a = my_rows;
                    for (uint32_t i = 1; i < w; ++i) {
                        const uint32_t ri = base + (i < extra ? 1u : 0u);
                        ship(
                            in_addr,
                            get_noc_addr(wx(i), wy(i), get_write_ptr(cb_partials_gathered)),
                            r0 + a,
                            ri,
                            GROUP_SIZE,
                            my_slot,
                            0,
                            GATHER_FACES);
                        a += ri;
                    }
                    noc_async_write_barrier();  // data before ALL signals
                    for (uint32_t i = 1; i < w; ++i) {
                        sem1.up(noc, wx(i), wy(i), 1);
                    }
                }
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arr1 += GROUP_SIZE - 1;
                    sem1.wait_min(arr1);
                    cb_push_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_send");
                    const uint32_t my_rows = base + (0 < extra ? 1u : 0u);
                    cb_wait_front(cb_stat_handoff, my_rows);
                    cb_reserve_back(cb_row_final, rows);
                    const uint32_t dst = get_write_ptr(cb_row_final);
                    // my finished rows land at row 0 of the block; the other workers
                    // write theirs straight into this same buffer.
                    noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(dst), my_rows * stat_bytes);
                    noc_async_write_barrier();
                    arr2 += w - 1;
                    sem2.wait_min(arr2);
                    if constexpr (mc.active) {
                        sender.send(dst, dst, rows * stat_bytes);
                    }
                    cb_push_back(cb_row_final, rows);
                    cb_pop_front(cb_stat_handoff, my_rows);
                }
            }
        } else {
            auto receiver = mc.receiver(noc);
            const uint32_t rx = mc.sender_x();
            const uint32_t ry = mc.sender_y();
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                const uint32_t w = (W_MAX < rows) ? W_MAX : rows;
                const uint32_t base = rows / w, extra = rows % w;
                const bool am_worker = (my_slot < w);
                uint32_t my_a = 0, my_rows = 0;
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    if (am_worker) {
                        cb_reserve_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
                    }
                    uint32_t a = 0;
                    for (uint32_t i = 0; i < w; ++i) {
                        const uint32_t ri = base + (i < extra ? 1u : 0u);
                        if (i == my_slot) {
                            my_a = a;
                            my_rows = ri;
                            ship(
                                in_addr,
                                get_noc_addr(get_write_ptr(cb_partials_gathered)),
                                r0 + a,
                                ri,
                                GROUP_SIZE,
                                my_slot,
                                0,
                                GATHER_FACES);
                        } else {
                            ship(
                                in_addr,
                                get_noc_addr(wx(i), wy(i), get_write_ptr(cb_partials_gathered)),
                                r0 + a,
                                ri,
                                GROUP_SIZE,
                                my_slot,
                                0,
                                GATHER_FACES);
                        }
                        a += ri;
                    }
                    noc_async_write_barrier();
                    for (uint32_t i = 0; i < w; ++i) {
                        if (i != my_slot) {
                            sem1.up(noc, wx(i), wy(i), 1);
                        }
                    }
                }
                if (am_worker) {
                    {
                        MaybeDeviceZoneScope("writer_gather_wait");
                        arr1 += GROUP_SIZE - 1;
                        sem1.wait_min(arr1);
                        cb_push_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
                    }
                    {
                        // Write my FINISHED stat rows straight into the ROOT's
                        // cb_row_final (the mcast source) at my row offset, then
                        // signal.  Every core pushes `rows` per round, so this
                        // core's get_write_ptr(cb_row_final) IS the root's.
                        MaybeDeviceZoneScope("writer_stage2_ship");
                        cb_wait_front(cb_stat_handoff, my_rows);
                        const uint32_t dst = get_write_ptr(cb_row_final) + my_a * stat_bytes;
                        noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(rx, ry, dst), my_rows * stat_bytes);
                        noc_async_write_barrier();  // data before signal
                        sem2.up(noc, rx, ry, 1);
                        cb_pop_front(cb_stat_handoff, my_rows);
                    }
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
}
