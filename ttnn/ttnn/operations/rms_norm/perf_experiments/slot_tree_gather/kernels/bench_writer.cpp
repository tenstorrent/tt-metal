// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/slot_tree_gather) -- NOT the op.
//
// The rms_norm cross-core width COMBINE, and nothing else.  Every core of a group starts
// with `num_rows` fp32 partial tiles already resident in its own L1 (a HEIGHT-sharded fp32
// input tensor -- pass A is deliberately not modelled), and every core must end with the
// group's finalized stat rsqrt(sum_group(partial) * INV_W + eps) in cb_row_final, which is
// backed on the OUTPUT shard -- so the result IS the output tensor.
//
//   VARIANT 0  FLAT  the op's CURRENT approach, carried verbatim (the honest baseline):
//                    every member ships its partial into its own slot of the single ROOT's
//                    gather CB (row-major landing: page = r * GATHER_SLOTS + my_slot, D16 +
//                    D22) and remote-incs the root's arrival semaphore; the root folds
//                    GATHER_SLOTS partials per row and finalizes them in ONE DEST window
//                    (D22), then multicasts the stat back (D24 ordering).
//
//   VARIANT 1  TREE  a k-ary tree over the SLOT axis, m == 1 (NO row split -- that half is
//                    exclusive with compact_partial_transpose; this half is not).  L levels
//                    of contiguous slot chunks with arity list F = (f0..f_{L-1}):
//                      stride[0] = 1, stride[l+1] = stride[l] * f_l;
//                      PARTICIPANT at level l  iff  slot % stride[l]     == 0;
//                      GATHERER    at level l  iff  slot % stride[l + 1] == 0;
//                      the chunk gathered by core p at level l is
//                        { p + j*stride[l] : j < f_l },
//                      real member count = min(f_l, ceil((GROUP_SIZE - p) / stride[l])).
//                    prod(F) >= GROUP_SIZE is asserted, so slot 0 -- the multicast root --
//                    is the UNIQUE gatherer at the last level and the multicast is
//                    completely untouched by the tree.  A level-l gatherer folds its chunk,
//                    packs the RAW sum (no finalize) and forwards it at level l+1; only the
//                    last level's fold runs the rsqrt finalize.
//
// EVEN SLOTS PER LEVEL.  Each level's ring is `f_l` rounded UP TO EVEN, exactly as D22 does
// for GATHER_SLOTS, so every fold is a pairwise DEST walk with an even count to halve.  Pad
// slots (the evenness slot, and any slot a RAGGED chunk has no real member for) are
// boot-zeroed WHOLE and pair against a real contributor as an exact +0.0.  That is what
// makes ONE code path cover GROUP_SIZE = 9 and GROUP_SIZE = 28 with no guard.
//
// RING DISCIPLINE.  Every gather ring is sized for ONE gatherer's share -- SLOTS_l *
// BLOCK_ROWS pages -- and the gatherer pushes/pops the WHOLE ring every round, so
// `get_write_ptr` returns the ring BASE at the start of every round on every core.  That is
// what lets a sender compute the landing address LOCALLY (the CB is declared identically on
// every core, so its L1 address is identical everywhere) and keeps the host out of CB
// addresses.
//
// FLOW CONTROL is the multicast, transitively.  A remote sender does not cb_reserve, so
// nothing but the previous round's mcast stops it overwriting a ring its gatherer is still
// folding.  It is safe because the root only sends round `blk`'s stat after its OWN
// last-level fold, which requires every level-(L-1) forward, which requires every
// level-(L-2) gatherer to have drained its ring, and so on down to level 0.  The op relies
// on the identical argument for its single root.
//
// NO SELF-SIGNAL, ANYWHERE.  `Semaphore::up(value)` is a NON-ATOMIC local read-modify-write
// (noc_semaphore.h: "multiple cores incrementing simultaneously may lead to lost updates"),
// so a local bump on a gatherer would race the members' remote atomic incs and silently
// drop one -- a HANG.  A tree makes this sharper than the flat op does, because an interior
// node is BOTH a receiver and a sender: it writes its own slot synchronously at every level
// it gathers and waits for exactly the OTHER (cnt - 1) contributors, and it signals only
// upward, never to itself.
//
// ONE SEMAPHORE PER LEVEL, not one shared counter.  A level-(l+1) sender only has to finish
// its OWN level-l chunk first, which is a DIFFERENT chunk from its parent's -- so it can
// legally arrive before one of the parent's level-l members.  A single cumulative counter
// would let that early level-(l+1) inc satisfy the parent's level-l wait_min and fold a
// slot that has not landed.  Per-level semaphores make the two waits independent.

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
constexpr uint32_t cb_gather0 = 11;       // level-l gather ring is cb_gather0 + l
constexpr uint32_t cb_node_out = 15;      // an interior gatherer's folded (NOT finalized) sum
constexpr uint32_t cb_stat_handoff = 16;  // finalized stat, compute -> writer
constexpr uint32_t cb_row_final = 17;     // mcast landing == the output shard
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_LEVELS = get_compile_time_arg_val(3);
    constexpr uint32_t F0 = get_compile_time_arg_val(4);
    constexpr uint32_t F1 = get_compile_time_arg_val(5);
    constexpr uint32_t F2 = get_compile_time_arg_val(6);
    constexpr uint32_t F3 = get_compile_time_arg_val(7);
    constexpr uint32_t SEM_BASE = get_compile_time_arg_val(8);
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(9);
    constexpr uint32_t RT_MC_BASE = 4 + 2 * GROUP_SIZE;
    constexpr auto mc = dataflow_kernel_lib::McastArgs<10, RT_MC_BASE>();

    // D22's landing stride for the FLAT baseline: GROUP_SIZE rounded up to even.
    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;

    constexpr uint32_t FA[4] = {F0, F1, F2, F3};
    constexpr uint32_t SL[4] = {F0 + F0 % 2, F1 + F1 % 2, F2 + F2 % 2, F3 + F3 % 2};
    constexpr uint32_t ST[5] = {1, F0, F0 * F1, F0 * F1 * F2, F0 * F1 * F2 * F3};
    static_assert(NUM_LEVELS >= 1 && NUM_LEVELS <= 4, "NUM_LEVELS must be 1..4");
    static_assert(VARIANT == 0 || ST[NUM_LEVELS] >= GROUP_SIZE, "TREE: prod(F) must cover GROUP_SIZE");
    static_assert(GATHER_FACES == 2 || GATHER_FACES == 3 || GATHER_FACES == 4, "GATHER_FACES must be 2, 3 or 4");

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
    Semaphore<> sems[4] = {
        Semaphore<>(SEM_BASE + 0), Semaphore<>(SEM_BASE + 1), Semaphore<>(SEM_BASE + 2), Semaphore<>(SEM_BASE + 3)};

    // Slot-indexed virtual-coord table of the whole group (RT 4 + 2*slot).
    auto vx = [&](uint32_t slot) { return get_arg_val<uint32_t>(4 + 2 * slot); };
    auto vy = [&](uint32_t slot) { return get_arg_val<uint32_t>(4 + 2 * slot + 1); };

    // ---- ONE definition of the partial transfer ------------------------------
    // `src` + `src_row0` index the SOURCE by tile-row; `dst_base` is the destination ring's
    // BASE and the (row, slot) offset is applied here.  `faces == 2` ships only faces 0 and
    // 2 (the pair that can hold a REDUCE_ROW column vector) -- half the bytes, two
    // transactions.  Identical to the op's ship_partial.
    auto ship = [&](uint32_t src,
                    uint32_t src_row0,
                    uint64_t dst_base,
                    uint32_t rows,
                    uint32_t dst_stride,
                    uint32_t dst_slot,
                    uint32_t faces) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t s_off = (src_row0 + r) * stat_bytes;
            const uint32_t d_off = (r * dst_stride + dst_slot) * stat_bytes;
            if (faces == 4) {
                noc_async_write(src + s_off, dst_base + d_off, stat_bytes);
            } else if (faces == 3) {
                noc_async_write(src + s_off, dst_base + d_off, 3 * face_bytes);
            } else {
                noc_async_write(src + s_off, dst_base + d_off, face_bytes);
                noc_async_write(src + s_off + 2 * face_bytes, dst_base + d_off + 2 * face_bytes, face_bytes);
            }
        }
    };

    // Boot: make every byte a fold can read DEFINED, so no undefined L1 reaches the
    // pairwise add / rsqrt.
    //   * a PAD slot (evenness slot, or a ragged chunk's missing member) is never written by
    //     anybody, so it is zeroed WHOLE and contributes an exact +0.0;
    //   * a REAL slot has only the faces the sender ships written, so exactly the UNSHIPPED
    //     faces are zeroed.  Zeroing the whole ring instead would race a member's partial
    //     that already landed (the op records that as pcc 0.87-0.99 on every combine cell).
    auto zero_ring = [&](uint32_t cb, uint32_t slots, uint32_t real_cnt, uint32_t faces) {
        MaybeDeviceZoneScope("writer_gather_zero");
        DataflowBuffer dfb(cb);
        const uint32_t pages = dfb.get_total_size_bytes() / stat_bytes;
        for (uint32_t p = 0; p < pages; ++p) {
            const uint32_t base = p * stat_bytes;
            if (p % slots >= real_cnt) {
                noc.async_write_zeros(dfb, stat_bytes, {.offset_bytes = base});
                continue;
            }
            if (faces == 2) {
                noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + face_bytes});
            }
            if (faces < 4) {
                noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes});
            }
        }
        noc.write_zeros_l1_barrier();
    };

    // The mcast back is identical in both variants: the root places its OWN copy first,
    // PUBLISHES it (D24 -- so its own pass B is not stuck behind the broadcast) and
    // broadcasts IN PLACE (src == dst => EXCLUDE-source), which is what makes Mcast1D's
    // per-row rect (excludes the sender) and Mcast2D's rect (contains it) behave the same.

    if constexpr (VARIANT == 0) {
        // ================= FLAT (the op's current approach, verbatim) =============
        if (is_root != 0) {
            if constexpr (GATHER_FACES < 4 || GATHER_SLOTS != GROUP_SIZE) {
                zero_ring(cb_gather0, GATHER_SLOTS, GROUP_SIZE, GATHER_FACES);
            }
            auto sender = mc.sender(noc);
            uint32_t arrivals = 0;
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_reserve_back(cb_gather0, GATHER_SLOTS * rows);
                    ship(
                        in_addr,
                        r0,
                        get_noc_addr(get_write_ptr(cb_gather0)),
                        rows,
                        GATHER_SLOTS,
                        my_slot,
                        GATHER_FACES);
                    noc_async_write_barrier();
                }
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arrivals += GROUP_SIZE - 1;
                    sems[0].wait_min(arrivals);
                    cb_push_back(cb_gather0, GATHER_SLOTS * rows);
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_send");
                    cb_wait_front(cb_stat_handoff, rows);
                    cb_reserve_back(cb_row_final, rows);
                    const uint32_t dst = get_write_ptr(cb_row_final);
                    noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(dst), rows * stat_bytes);
                    noc_async_write_barrier();
                    cb_push_back(cb_row_final, rows);
                    if constexpr (mc.active) {
                        sender.send(dst, dst, rows * stat_bytes);
                    }
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
                        r0,
                        get_noc_addr(rx, ry, get_write_ptr(cb_gather0)),
                        rows,
                        GATHER_SLOTS,
                        my_slot,
                        GATHER_FACES);
                    noc_async_write_barrier();  // data before signal
                    sems[0].up(noc, rx, ry, 1);
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

    // ================= TREE (L levels of contiguous slot chunks, m == 1) ==========
    // How many REAL members the level-l chunk gathered by `p` has.
    auto chunk_cnt = [&](uint32_t l, uint32_t p) {
        const uint32_t stride = ST[l];
        const uint32_t span = (GROUP_SIZE - p + stride - 1) / stride;
        return (FA[l] < span) ? FA[l] : span;
    };

    // Boot-zero every ring I will ever fold from.  `slot % ST[l+1] == 0` is the gatherer
    // predicate; strides are increasing multiples, so the first miss ends it.
    for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
        if (my_slot % ST[l + 1] != 0) {
            break;
        }
        const uint32_t faces = (l == 0) ? GATHER_FACES : 4;
        zero_ring(cb_gather0 + l, SL[l], chunk_cnt(l, my_slot), faces);
    }

    uint32_t arr[4] = {0, 0, 0, 0};

    // Stages are identical on every core, so the tree walk lives in one lambda and the
    // root / non-root split below only owns the mcast tail (whose pipe CONSTRUCTOR is the
    // documented handshake init and therefore must not run on the wrong role).
    auto walk = [&](uint32_t r0, uint32_t rows) {
        uint32_t src = in_addr;  // level 0 reads this core's own resident partial shard
        uint32_t src_row0 = r0;  // ... indexed by ABSOLUTE tile-row
        bool from_cb = false;    // ... every later level reads cb_node_out from row 0
        for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
            const uint32_t cb = cb_gather0 + l;
            const uint32_t slots = SL[l];
            const uint32_t parent = (my_slot / ST[l + 1]) * ST[l + 1];
            const uint32_t pos = (my_slot - parent) / ST[l];
            const bool mine = (parent == my_slot);
            const uint32_t faces = (l == 0) ? GATHER_FACES : 4;
            {
                MaybeDeviceZoneScope("writer_gather_ship");
                if (from_cb) {
                    // Faces 1/3 of a folded tile are the ring's boot zeros carried through
                    // DEST, so an interior forward is exact as a WHOLE-tile transfer and
                    // needs no second zeroing pass -- one transaction per row.
                    cb_wait_front(cb_node_out, rows);
                    src = get_read_ptr(cb_node_out);
                    src_row0 = 0;
                }
                if (mine) {
                    cb_reserve_back(cb, slots * BLOCK_ROWS);
                }
                const uint32_t wp = get_write_ptr(cb);
                const uint64_t dst = mine ? get_noc_addr(wp) : get_noc_addr(vx(parent), vy(parent), wp);
                ship(src, src_row0, dst, rows, slots, pos, faces);
                noc_async_write_barrier();  // data before signal
                if (!mine) {
                    sems[l].up(noc, vx(parent), vy(parent), 1);  // NEVER to myself
                }
                if (from_cb) {
                    cb_pop_front(cb_node_out, rows);
                }
            }
            if (!mine) {
                return;  // I contributed at this level and am not its gatherer: done.
            }
            {
                MaybeDeviceZoneScope("writer_gather_wait");
                arr[l] += chunk_cnt(l, my_slot) - 1;  // my own slot was written synchronously
                sems[l].wait_min(arr[l]);
                cb_push_back(cb, slots * BLOCK_ROWS);
            }
            from_cb = true;
        }
    };

    if (is_root != 0) {
        auto sender = mc.sender(noc);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t r0 = blk * BLOCK_ROWS;
            const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
            walk(r0, rows);
            MaybeDeviceZoneScope("writer_mcast_send");
            cb_wait_front(cb_stat_handoff, rows);
            cb_reserve_back(cb_row_final, rows);
            const uint32_t dst = get_write_ptr(cb_row_final);
            noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(dst), rows * stat_bytes);
            noc_async_write_barrier();
            cb_push_back(cb_row_final, rows);
            if constexpr (mc.active) {
                sender.send(dst, dst, rows * stat_bytes);
            }
            cb_pop_front(cb_stat_handoff, rows);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t r0 = blk * BLOCK_ROWS;
            const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
            walk(r0, rows);
            {
                MaybeDeviceZoneScope("writer_mcast_recv");
                cb_reserve_back(cb_row_final, rows);
                receiver.receive();
                cb_push_back(cb_row_final, rows);
            }
        }
    }
}
