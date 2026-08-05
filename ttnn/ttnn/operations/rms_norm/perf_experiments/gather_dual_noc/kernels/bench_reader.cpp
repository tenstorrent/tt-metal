// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/gather_dual_noc) -- NOT the op.
//
// THE READER HALF of the dual-NoC combine.  In the op today this kernel is essentially
// EMPTY on the focus shape (native zero-copy L1 shard: `reader_read_x` = 56 ns for the
// whole kernel, NCRISC / NoC0 idle for ~31 us) while BRISC / NoC1 issues every byte and
// every synchronization of the cross-core combine.  This bench moves work here.
//
// What this kernel does, per compile-time knob (all zero == the op's CURRENT approach, i.e.
// this kernel only runs its synthetic NoC0 load and returns):
//
//   RD_TILES     synthetic DRAM read load, `RD_TILES` bf16 tiles per round, issued BEFORE
//                this kernel's share of the ship -- the op's dependency order (stage x for
//                the block, then ship the block's partial).  RD_TILES == 0 models the
//                native-shard focus case (NoC0 idle); a large RD_TILES models the
//                reader-fed INTERLEAVED case where NoC0 is emphatically NOT idle.
//   ZERO_R       the root's one-time `writer_gather_zero` boot moves here (NoC0).
//   SPLIT_MODE   1 ROWS  : this kernel ships rows [0, rows*NUM/DEN) of the partial
//                2 FACES : this kernel ships FACE 0 of every row (the writer takes face 2)
//                3 ALL   : this kernel ships the whole partial
//   MCAST_R      the stat multicast moves here -- SenderPipe on the root, ReceiverPipe on
//                every member, both on NoC0.
//
// ---------------------------------------------------------------------------------------
// THE HAPPENS-BEFORE EDGES (the trap in this idea, built explicitly)
// ---------------------------------------------------------------------------------------
// `noc_async_write_barrier()` on NCRISC flushes only NoC0's outstanding writes; the
// writer's barrier flushes only NoC1's.  Neither sees the other.  And
// `Semaphore::up(value)` is a NON-ATOMIC local read-modify-write, so a member's arrival
// signal must be raised EXACTLY ONCE and only after BOTH halves have landed -- a lost
// update is a hang, a premature one silently folds a TORN partial.  So every edge between
// the two dataflow kernels ON THE SAME CORE is a single-producer/single-consumer token CB:
//
//   tok_w2r  writer -> reader   "GO": the gather ring is reserved / the source is readable.
//                               Also the flow control: on the root it is pushed AFTER
//                               cb_reserve_back(cb_partials_gathered) so this kernel can
//                               never write into a ring compute has not drained; on a
//                               member the writer's round is itself gated by the previous
//                               round's mcast (the op's own transitive-flow-control
//                               argument), so the GO inherits that gate.
//   tok_r2w  reader -> writer   three signals, pushed in this order and consumed by the
//                               writer in the SAME order (one FIFO, so it cannot reorder):
//                                 * ZERO_R  : once, "the unshipped faces are zeroed".  The
//                                             writer consumes it before its FIRST
//                                             cb_push_back(cb_partials_gathered) -- the
//                                             push is what releases the ring to compute,
//                                             so that is the only place it must be visible.
//                                 * SPLIT   : per round, "my NoC0 half has LANDED" (after
//                                             my barrier).  The writer raises the ONE
//                                             semaphore inc after this.
//                                 * MCAST_R : per round, "my mcast round is done".  A
//                                             member's writer waits for the PREVIOUS
//                                             round's copy at the top of the next round --
//                                             it has to, because moving the receive here
//                                             removes the writer's only gate and an
//                                             ungated member would ship round blk+1 into a
//                                             ring the root is still folding.
//
// The gather landing address is `get_write_ptr(cb_partials_gathered)` computed LOCALLY:
// that CB is declared on every core, so its L1 address is identical everywhere, and its
// ring holds exactly GROUP_SIZE * BLOCK_ROWS pages, so a whole-block push returns the
// pointer to the base every round -- this kernel never pushes it at all, so its (never
// advanced) write pointer IS that base.  Exactly the op's argument, reused.

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
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_stat_handoff = 15;
constexpr uint32_t cb_row_final = 16;
constexpr uint32_t cb_tok_w2r = 17;
constexpr uint32_t cb_tok_r2w = 18;
constexpr uint32_t cb_tok_zero = 20;  // the ZERO_R edge has its OWN FIFO
constexpr uint32_t cb_load = 19;

constexpr uint32_t SPLIT_NONE = 0;
constexpr uint32_t SPLIT_ROWS = 1;
constexpr uint32_t SPLIT_FACES = 2;
constexpr uint32_t SPLIT_ALL = 3;
}  // namespace

void kernel_main() {
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);
    constexpr uint32_t SEM1 = get_compile_time_arg_val(2);
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(3);
    constexpr uint32_t ZERO_R = get_compile_time_arg_val(4);
    constexpr uint32_t SPLIT_MODE = get_compile_time_arg_val(5);
    constexpr uint32_t SPLIT_NUM = get_compile_time_arg_val(6);
    constexpr uint32_t SPLIT_DEN = get_compile_time_arg_val(7);
    constexpr uint32_t MCAST_R = get_compile_time_arg_val(8);
    constexpr uint32_t TOK_ROUND = get_compile_time_arg_val(9);
    // SPLIT_ROOT == 0 means only the MEMBERS split their ship; the root keeps its own
    // (purely LOCAL) slot write entirely on the writer.  This is not a cosmetic knob -- see
    // the note on the root's `landed` token below.
    constexpr uint32_t SPLIT_ROOT = get_compile_time_arg_val(10);
    constexpr uint32_t RD_TILES = get_compile_time_arg_val(11);
    constexpr uint32_t LOAD_TILES = get_compile_time_arg_val(12);
    constexpr auto mc = dataflow_kernel_lib::McastArgs<13, 5>();
    constexpr auto load_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    static_assert(GATHER_FACES == 2, "this bench pins the op's compact 2-face gather");
    static_assert(SPLIT_MODE <= SPLIT_ALL, "SPLIT_MODE must be 0..3");

    constexpr bool R_SHIPS = (SPLIT_MODE != SPLIT_NONE);
    // The token ping-pong is compiled in when the split needs it OR as the pure-overhead
    // control (`tok`): the writer then still does the whole ship and the ONLY difference
    // from `base` is the edge itself.
    constexpr bool TOKENS = (TOK_ROUND != 0);
    // MCAST_R: 0 = the whole multicast stays on the writer.  1 = BOTH pipe faces here.
    // 2 = only the ROOT's SEND here; the members keep receiving on the writer, so they keep
    // their own round gate and mode 2 needs NO new edge at all.
    constexpr bool MC_SEND_HERE = (MCAST_R != 0);
    constexpr bool MC_RECV_HERE = (MCAST_R == 1);

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t my_slot = get_arg_val<uint32_t>(3);
    const uint32_t load_addr = get_arg_val<uint32_t>(4);

    // An INACTIVE core: it joined the program only so the stat multicast lands in a
    // cb_row_final this program owns.  No shard, no work, no ack.
    if (num_rows == 0) {
        return;
    }

    const uint32_t stat_bytes = get_tile_size(cb_stat_handoff);
    const uint32_t face_bytes = stat_bytes / 4;
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    Noc noc;

    // ---- ONE definition of the partial transfer, shared with the writer -------------
    // `src` is indexed by ABSOLUTE tile-row inside this core's resident partial shard;
    // `dst_base` is the root gather ring's BASE (row/slot offset applied here).
    // `faces` is a MASK: bit0 = face 0, bit1 = face 2 (the pair that can hold a REDUCE_ROW
    // column vector -- the op's compact 2-face gather).
    auto ship = [&](uint64_t dst_base, uint32_t abs_row0, uint32_t dst_row0, uint32_t rows, uint32_t faces) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t s_off = (abs_row0 + r) * stat_bytes;
            const uint32_t d_off = ((dst_row0 + r) * GROUP_SIZE + my_slot) * stat_bytes;
            if (faces & 0x1u) {
                noc_async_write(in_addr + s_off, dst_base + d_off, face_bytes);
            }
            if (faces & 0x2u) {
                noc_async_write(in_addr + s_off + 2 * face_bytes, dst_base + d_off + 2 * face_bytes, face_bytes);
            }
        }
    };

    // ---- the synthetic NoC0 load: what a reader-FED placement makes this kernel do ----
    auto reader_load = [&](uint32_t blk) {
        if constexpr (RD_TILES > 0) {
            MaybeDeviceZoneScope("reader_load");
            const auto acc = TensorAccessor(load_args, load_addr);
            const uint32_t load_tile_bytes = get_tile_size(cb_load);
            const uint32_t base = get_write_ptr(cb_load);
            for (uint32_t i = 0; i < RD_TILES; ++i) {
                const uint32_t id = (my_slot * 37u + blk * RD_TILES + i) % LOAD_TILES;
                noc_async_read_tile(id, acc, base + (i & 0x3u) * load_tile_bytes);
            }
            noc_async_read_barrier();
        }
    };

    // ---- ZERO_R: the root's gather-zero boot, on NoC0 --------------------------------
    // Zeroing EXACTLY the unshipped faces (1 and 3) is what makes this race-free: a
    // member's partial can land at any time and only ever touches faces 0 and 2, so the
    // boot and the gather are byte-disjoint and their order does not matter.  Zeroing the
    // whole ring instead wipes members that already arrived (the op measured pcc
    // 0.87-0.99).  Moving it to NoC0 does not touch that argument -- only WHO issues it.
    if constexpr (ZERO_R != 0) {
        if (is_root != 0) {
            {
                MaybeDeviceZoneScope("reader_gather_zero");
                DataflowBuffer dfb(cb_partials_gathered);
                const uint32_t pages = dfb.get_total_size_bytes() / stat_bytes;
                for (uint32_t p = 0; p < pages; ++p) {
                    const uint32_t base = p * stat_bytes;
                    noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + face_bytes});
                    noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes});
                }
                noc.write_zeros_l1_barrier();
            }
            cb_reserve_back(cb_tok_zero, 1);
            cb_push_back(cb_tok_zero, 1);  // "the unshipped faces are zeroed"
        }
    }

    // ---- my share of the ship --------------------------------------------------------
    // ROWS  : rows [0, rows*NUM/DEN)  (FLOOR -- at rows == 1 the reader's share is 0 rows,
    //         which is the honest answer for a BLOCK_ROWS = 1 round: there is nothing to
    //         split by row, only the token edge remains, and `sf` is the mode that CAN
    //         split it).
    // FACES : every row, face 0 only.
    // ALL   : every row, both faces.
    auto my_share_rows = [&](uint32_t rows) -> uint32_t {
        if constexpr (SPLIT_MODE == SPLIT_ROWS) {
            return (rows * SPLIT_NUM) / SPLIT_DEN;
        } else if constexpr (SPLIT_MODE == SPLIT_NONE) {
            return 0;
        } else {
            return rows;
        }
    };
    constexpr uint32_t MY_FACES = (SPLIT_MODE == SPLIT_FACES) ? 0x1u : 0x3u;

    const uint32_t root_x = mc.sender_x();
    const uint32_t root_y = mc.sender_y();

    // ---- the per-round body, with the mcast step passed in ---------------------------
    // (the mcast step is a callable so the pipe OBJECT can live outside the loop -- a
    // ReceiverPipe ctor kernel-inits its own data_ready cell and must run EXACTLY ONCE per
    // core per program, so it can neither be re-constructed per round nor constructed on
    // both dataflow kernels.)
    // WHY SPLIT_ROOT MATTERS (measured).  The root's `landed` token is a WRITER-WAITS-ON-
    // READER edge, and on the root this kernel ALSO owns the multicast send under MCAST_R.
    // Since a round here is [ship my share] -> [mcast send], the writer's round blk+1
    // `landed` wait then transitively waits for the mcast send of round blk -- re-coupling
    // exactly the two stages the mcast move had just decoupled.  Measured on the focus
    // geometry (fold-ablated): mcast-send-only 13488 ns, + a 50/50 row split ON THE ROOT TOO
    // 18897 ns -- i.e. composing two individual WINS produced a LOSS.  The root's own slot
    // write is a purely LOCAL L1 write and the cheapest part of the gather, so SPLIT_ROOT = 0
    // gives it up and keeps the decoupling.
    const bool split_here = (SPLIT_ROOT != 0) || (is_root == 0);

    auto round_body = [&](uint32_t blk, auto&& mcast_step) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        reader_load(blk);
        if constexpr (TOKENS) {
            if (split_here) {
                MaybeDeviceZoneScope("reader_gather_ship");
                cb_wait_front(cb_tok_w2r, 1);  // GO (and the flow control, transitively)
                cb_pop_front(cb_tok_w2r, 1);
                if constexpr (R_SHIPS) {
                    const uint32_t n = my_share_rows(rows);
                    if (n != 0) {
                        const uint64_t dst = (is_root != 0)
                                                 ? get_noc_addr(get_write_ptr(cb_partials_gathered))
                                                 : get_noc_addr(root_x, root_y, get_write_ptr(cb_partials_gathered));
                        ship(dst, r0, /*dst_row0=*/0, n, MY_FACES);
                    }
                    noc_async_write_barrier();  // MY NoC0 half has landed
                }
                cb_reserve_back(cb_tok_r2w, 1);
                cb_push_back(cb_tok_r2w, 1);
            }
        }
        mcast_step(rows);
    };

    if constexpr (MC_SEND_HERE) {
        if (is_root != 0) {
            auto sender = mc.sender(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                round_body(blk, [&](uint32_t rows) {
                    MaybeDeviceZoneScope("reader_mcast_send");
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
                });
            }
        } else if constexpr (MC_RECV_HERE) {
            auto receiver = mc.receiver(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                round_body(blk, [&](uint32_t rows) {
                    {
                        MaybeDeviceZoneScope("reader_mcast_recv");
                        cb_reserve_back(cb_row_final, rows);
                        receiver.receive();
                        cb_push_back(cb_row_final, rows);
                    }
                    // The writer lost its only gate when the receive moved here; hand it
                    // back so an ungated member cannot ship the NEXT round into a ring the
                    // root is still folding.  This is the cost that makes MCAST_R == 1 lose
                    // to MCAST_R == 2, where the member keeps its own receive and its gate.
                    cb_reserve_back(cb_tok_r2w, 1);
                    cb_push_back(cb_tok_r2w, 1);
                });
            }
        } else {
            // MCAST_R == 2 on a MEMBER: the receive stayed on the writer, so this kernel has
            // only its split share (if any) to do.
            if constexpr (TOKENS || RD_TILES > 0) {
                for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                    round_body(blk, [](uint32_t) {});
                }
            }
        }
    } else {
        // The mcast stays on the writer.  Without a split and without a load this kernel
        // is what the op's reader is today on the focus shape: nothing at all.
        if constexpr (TOKENS || RD_TILES > 0) {
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                round_body(blk, [](uint32_t) {});
            }
        }
    }

    // KERNEL-EXIT HYGIENE, in EVERY variant so it cannot bias the comparison.  A member's
    // last act is `Semaphore::up`, a NON-POSTED ATOMIC, and nothing after it flushes the
    // atomic counter: leaving one in flight at kernel exit corrupts the next kernel's NoC
    // bookkeeping, and BRISC's post-kernel
    // ASSERT(ncrisc_noc_nonposted_atomics_flushed) halts the core under --dev (measured:
    // the mcast-on-reader variant hung here with the whole group's kernels already
    // finished).  The baseline only got away without it because its last act was the
    // multicast receive, whose ack the pipe flushes itself.
    noc.async_full_barrier();
}
