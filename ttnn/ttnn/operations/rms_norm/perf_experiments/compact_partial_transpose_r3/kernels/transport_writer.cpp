// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/compact_partial_transpose_r3) -- NOT the op's kernel.
//
// BENCH B: the WHOLE cross-core combine (transport + semaphore sync + multicast + the root's
// fold), FLAT vs COMPACT.  Bench A (combine_bench.py) measures the root's COMPUTE in isolation;
// this bench exists because the compact layout also collapses the TRANSPORT, and the Perf-3 peel
// says the transport + sync residue is 46.7% of the whole op -- bigger than the fold.
//
// THE TRANSPORT NUMBER THIS BENCH IS BUILT TO MEASURE.  At the focus geometry the op ships a
// partial with GATHER_FACES == 2: TWO 1024 B NoC writes per tile-row per member, i.e. 16 writes /
// 16 kB per member per round, to carry 8 rows x 32 fp32 = 1 kB of actual information -- 16x byte
// amplification in sub-face-sized chunks.  Under COMPACT a member ships ONE WHOLE 4096 B tile:
// one transaction, 4 kB, carrying the same 1 kB.  The multicast back shrinks by the same factor
// (1 tile instead of BLOCK_ROWS).
//
// WHY COMPACT SHIPS WHOLE TILES AND THEREFORE NEEDS NO BOOT-ZEROING.  The op's GATHER_FACES < 4
// gather leaves the unshipped faces of every landing page UNDEFINED, which is why the writer has
// a `writer_gather_zero` boot stage at all.  A compact page cannot do that: the receiver's
// un-permute is a matmul, which sums 32 products across the row, so an inf/NaN bit pattern in ANY
// column becomes inf*0 = NaN in column 0.  Shipping the WHOLE tile makes every byte of every
// landing page written by exactly one member per round -- defined by construction, one
// transaction, and `writer_gather_zero` disappears (except for the odd-GROUP_SIZE pad, which is
// now ONE page instead of GATHER_SLOTS * BLOCK_ROWS).
//
// FLAT is the op's CURRENT writer (kernels/rms_norm_writer.cpp), carried verbatim: the row-major
// landing layout at stride GATHER_SLOTS (D16 + D22), the GATHER_FACES == 2 two-transaction ship
// (D13), the boot-zeroing of exactly the unshipped faces (race-free because a member only ever
// writes faces the root leaves alone), NO self-signal on the root (Semaphore::up is a non-atomic
// local RMW), and D24's publish-own-copy-before-the-broadcast.
//
// Pass A, pass B and the output write-back are deliberately NOT modelled: every core starts with
// its `num_rows` fp32 partials already resident in its own L1 shard, and the program's whole
// duration IS the combine.  So the measured delta is attributable to the collective alone -- but
// it also means the numbers here are the combine's EXPOSED cost, with none of the op's D25
// pipeline overlap, which is stated in measured.txt rather than hidden.

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
constexpr uint32_t cb_x = 0;                  // resident fp32 partials (the shard)
constexpr uint32_t cb_bank = 1;               // one-hot permutation bank
constexpr uint32_t cb_sum_handoff = 2;        // COMPACT only: compute's packed compact partial
constexpr uint32_t cb_partials_gathered = 3;  // the root's landing ring
constexpr uint32_t cb_stat_handoff = 4;       // the root's finalized stat, compute -> writer
constexpr uint32_t cb_mcast_in = 5;           // COMPACT only: the multicast landing (1 page)
constexpr uint32_t cb_row_final = 6;          // == the output shard
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);  // 0 = FLAT, 1 = COMPACT
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t SEM1 = get_compile_time_arg_val(3);
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(4);  // FLAT only
    constexpr uint32_t RT_MC_BASE = 8;
    constexpr auto mc = dataflow_kernel_lib::McastArgs<5, RT_MC_BASE>();

    // The op's D22 landing stride, DERIVED from GROUP_SIZE exactly as both op kernels derive it.
    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;
    constexpr bool COMPACT = (VARIANT == 1);
    // Pages the landing ring is SIZED for.  This IS the L1 lever:
    // GATHER_SLOTS * BLOCK_ROWS -> GATHER_SLOTS, flat in BLOCK_ROWS.
    constexpr uint32_t GATHER_RING = COMPACT ? GATHER_SLOTS : (GATHER_SLOTS * BLOCK_ROWS);
    static_assert(GATHER_FACES == 2 || GATHER_FACES == 3 || GATHER_FACES == 4, "faces must be 2..4");
    static_assert(GATHER_RING >= GATHER_SLOTS, "the ring must hold at least one round's window");

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t my_slot = get_arg_val<uint32_t>(3);

    // An INACTIVE core joined the program only so the stat multicast lands in a cb_row_final this
    // program owns (the op's row-major-packed WIDTH-shard grid does exactly this).
    if (num_rows == 0) {
        return;
    }

    const uint32_t stat_bytes = get_tile_size(cb_stat_handoff);
    const uint32_t face_bytes = stat_bytes / 4;
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    Noc noc;
    Semaphore<> sem1(SEM1);
    uint32_t arrivals = 0;

    // ---- FLAT's ship_partial, VERBATIM from the op ---------------------------------------
    // One transfer per tile-row per shipped face run; landing page = r * GATHER_SLOTS + my_slot.
    auto ship_flat = [&](uint32_t src, uint64_t dst_noc, uint32_t abs_row0, uint32_t rows) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t s_off = (abs_row0 + r) * stat_bytes;
            const uint32_t d_off = (r * GATHER_SLOTS + my_slot) * stat_bytes;
            if constexpr (GATHER_FACES == 4) {
                noc_async_write(src + s_off, dst_noc + d_off, stat_bytes);
            } else if constexpr (GATHER_FACES == 3) {
                noc_async_write(src + s_off, dst_noc + d_off, 3 * face_bytes);
            } else {
                noc_async_write(src + s_off, dst_noc + d_off, face_bytes);
                noc_async_write(src + s_off + 2 * face_bytes, dst_noc + d_off + 2 * face_bytes, face_bytes);
            }
        }
    };

    // ---- COMPACT's ship: ONE whole tile into page `my_slot` -------------------------------
    auto ship_compact = [&](uint32_t src, uint64_t dst_noc) {
        noc_async_write(src, dst_noc + my_slot * stat_bytes, stat_bytes);
    };

    // ---- the boot zeroing, and what COMPACT does to it -----------------------------------
    // FLAT: exactly the UNSHIPPED faces of every one of GATHER_SLOTS * BLOCK_ROWS pages (plus a
    //   whole pad page at odd GROUP_SIZE).  Zeroing the whole ring would race a member's already
    //   landed partial -- the op records that as pcc 0.87-0.99.
    // COMPACT: every byte of every real page is written by one member per round, so the ONLY
    //   thing left is the odd-GROUP_SIZE pad page.  At even GROUP_SIZE (4/8/28/32, including the
    //   focus shape) this loop does NOTHING AT ALL.
    auto boot_zero = [&]() {
        if (is_root == 0) {
            return;  // only the root reads this CB, so only the root pays
        }
        MaybeDeviceZoneScope("writer_gather_zero");
        DataflowBuffer dfb(cb_partials_gathered);
        const uint32_t pages = dfb.get_total_size_bytes() / stat_bytes;
        bool any = false;
        for (uint32_t p = 0; p < pages; ++p) {
            const uint32_t base = p * stat_bytes;
            if (p % GATHER_SLOTS >= GROUP_SIZE) {  // a pad slot no member ever writes
                noc.async_write_zeros(dfb, stat_bytes, {.offset_bytes = base});
                any = true;
                continue;
            }
            if constexpr (!COMPACT && GATHER_FACES < 4) {
                if constexpr (GATHER_FACES == 2) {
                    noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + face_bytes});
                }
                noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes});
                any = true;
            }
        }
        if (any) {
            noc.write_zeros_l1_barrier();
        }
    };

    boot_zero();

    if (is_root != 0) {
        auto sender = mc.sender(noc);
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t r0 = blk * BLOCK_ROWS;
            const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
            const uint32_t stat_pages = COMPACT ? 1 : rows;
            // A RAGGED last block has rows < BLOCK_ROWS, so FLAT's window is per-ROUND, not the
            // ring size.  Every core computes it identically, which is what keeps a remote
            // sender's locally-computed landing address equal to the root's.
            const uint32_t window = COMPACT ? GATHER_SLOTS : (GATHER_SLOTS * rows);

            // 1. the root's own partial goes into its own slot of its own landing ring.
            {
                MaybeDeviceZoneScope("writer_gather_ship");
                cb_reserve_back(cb_partials_gathered, window);
                const uint64_t dst = get_noc_addr(get_write_ptr(cb_partials_gathered));
                if constexpr (COMPACT) {
                    cb_wait_front(cb_sum_handoff, 1);
                    ship_compact(get_read_ptr(cb_sum_handoff), dst);
                } else {
                    ship_flat(x_addr, dst, r0, rows);
                }
                noc_async_write_barrier();
                if constexpr (COMPACT) {
                    cb_pop_front(cb_sum_handoff, 1);
                }
            }
            // NO self-signal: Semaphore::up(value) is a non-atomic local read-modify-write and
            // would race the members' remote atomic incs (the op records that as a hang).
            {
                MaybeDeviceZoneScope("writer_gather_wait");
                arrivals += GROUP_SIZE - 1;
                sem1.wait_min(arrivals);
                cb_push_back(cb_partials_gathered, window);
            }
            // 2. multicast the finalized stat back.  D24: publish the root's OWN copy first, then
            //    broadcast IN PLACE (src == dst => EXCLUDE-source), which makes Mcast1D's per-row
            //    rect and Mcast2D's rect behave identically.
            {
                MaybeDeviceZoneScope("writer_mcast_send");
                constexpr uint32_t CB_LAND = COMPACT ? cb_mcast_in : cb_row_final;
                cb_wait_front(cb_stat_handoff, stat_pages);
                cb_reserve_back(CB_LAND, stat_pages);
                const uint32_t dst = get_write_ptr(CB_LAND);
                noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(dst), stat_pages * stat_bytes);
                noc_async_write_barrier();
                cb_push_back(CB_LAND, stat_pages);
                if constexpr (mc.active) {
                    sender.send(dst, dst, stat_pages * stat_bytes);
                }
                cb_pop_front(cb_stat_handoff, stat_pages);
            }
        }
    } else {
        auto receiver = mc.receiver(noc);
        const uint32_t rx = mc.sender_x();
        const uint32_t ry = mc.sender_y();
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t r0 = blk * BLOCK_ROWS;
            const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
            const uint32_t stat_pages = COMPACT ? 1 : rows;
            {
                MaybeDeviceZoneScope("writer_gather_ship");
                const uint64_t dst = get_noc_addr(rx, ry, get_write_ptr(cb_partials_gathered));
                if constexpr (COMPACT) {
                    cb_wait_front(cb_sum_handoff, 1);
                    ship_compact(get_read_ptr(cb_sum_handoff), dst);
                } else {
                    ship_flat(x_addr, dst, r0, rows);
                }
                noc_async_write_barrier();  // data before signal
                sem1.up(noc, rx, ry, 1);
                if constexpr (COMPACT) {
                    cb_pop_front(cb_sum_handoff, 1);
                }
            }
            {
                MaybeDeviceZoneScope("writer_mcast_recv");
                constexpr uint32_t CB_LAND = COMPACT ? cb_mcast_in : cb_row_final;
                cb_reserve_back(CB_LAND, stat_pages);
                receiver.receive();
                cb_push_back(CB_LAND, stat_pages);
            }
        }
    }
}
