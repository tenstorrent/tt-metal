// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH (perf_experiments/gather_dual_noc) -- NOT the op.
//
// THE WRITER HALF of the dual-NoC combine.  At every knob == 0 this kernel IS the op's
// current approach, carried verbatim from kernels/rms_norm_writer.cpp's CROSS_CORE block
// (and identical to perf_experiments/hierarchical_gather_r2's FLAT baseline), so the
// measured delta is against what the op does today and not against a strawman:
//
//   * the compact 2-face `ship_partial` -- ONE definition used by the root for its own
//     local slot and by every member for the remote one, landing at the ROW-MAJOR page
//     r * GROUP_SIZE + my_slot of the root's gather ring (D16);
//   * the `writer_gather_zero` boot that zeroes EXACTLY the unshipped faces (1 and 3) --
//     race-free precisely because it is byte-disjoint from every member's write, which is
//     why zeroing the whole ring instead does NOT work (measured pcc 0.87-0.99);
//   * NO self-signal on the root: `Semaphore::up(value)` is a NON-ATOMIC local
//     read-modify-write, so a local bump would race the members' remote atomic incs and a
//     lost update is a HANG.  The root writes its own slot synchronously and waits for the
//     other GROUP_SIZE - 1 members;
//   * the root places its own copy of the stat FIRST and broadcasts IN PLACE
//     (src == dst => EXCLUDE-source), so Mcast1D's per-row rect and Mcast2D's rect behave
//     identically.
//
// The knobs hand pieces of that to the READER kernel on NoC0.  See bench_reader.cpp for
// the token-CB happens-before contract; this side of it is:
//
//   ZERO_R    : consume the one-time "faces zeroed" token before the FIRST
//               cb_push_back(cb_partials_gathered) (the push is what releases the ring to
//               compute, so that is the only place the zeroing must be visible).
//   SPLIT_*   : ship only MY complement of the partial, then WAIT for the reader's "landed"
//               token, and only THEN raise the ONE arrival inc.  This kernel owns the
//               semaphore in every mode -- one definition, so no mode can signal twice or
//               signal a torn partial.
//   MCAST_R   : the multicast moved to the reader, which took this kernel's only gate with
//               it; a member therefore waits the reader's "mcast round done" token at the
//               top of the next round.
//
// The GO token is pushed on the root AFTER cb_reserve_back(cb_partials_gathered), so the
// reader can never write into a ring compute has not drained.  On a member the round is
// gated by the previous round's multicast, exactly as the op argues, and the GO inherits it.

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
constexpr uint32_t cb_tok_zero = 20;  // the ZERO_R edge has its OWN FIFO -- see below

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
    // SPLIT_ROOT == 0: only the MEMBERS split; the root keeps its own (purely local) slot
    // write here.  See bench_reader.cpp for the measured reason.
    constexpr uint32_t SPLIT_ROOT = get_compile_time_arg_val(10);
    constexpr auto mc = dataflow_kernel_lib::McastArgs<11, 4>();

    static_assert(GATHER_FACES == 2, "this bench pins the op's compact 2-face gather");
    static_assert(SPLIT_MODE <= SPLIT_ALL, "SPLIT_MODE must be 0..3");

    constexpr bool TOKENS = (TOK_ROUND != 0);
    // MCAST_R: 0 = the whole multicast stays here.  1 = BOTH pipe faces move to the reader.
    // 2 = only the ROOT's SEND moves (the members keep receiving here).  2 is the placement
    // that costs NOTHING in edges: a member that still owns its own receive still owns its
    // round gate, so no token has to hand it back.  Only mode 1 needs the round gate.
    constexpr bool MC_SEND_HERE = (MCAST_R == 0);
    constexpr bool MC_RECV_HERE = (MCAST_R != 1);
    constexpr bool NEED_ROUND_GATE = (MCAST_R == 1);
    // My face mask: everything unless the reader took face 0.
    constexpr uint32_t MY_FACES = (SPLIT_MODE == SPLIT_FACES) ? 0x2u : 0x3u;

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t my_slot = get_arg_val<uint32_t>(3);

    if (num_rows == 0) {
        return;  // INACTIVE core (see bench_reader.cpp)
    }

    const uint32_t stat_bytes = get_tile_size(cb_stat_handoff);
    const uint32_t face_bytes = stat_bytes / 4;
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    Noc noc;
    Semaphore<> sem1(SEM1);
    uint32_t arrivals = 0;

    // ONE definition of the partial transfer (the op's ship_partial, `faces` as a mask).
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

    // My complement of the split: rows [n, rows) under a ROW split, nothing under ALL,
    // every row (face 2 only) under a FACE split.
    auto my_row0 = [&](uint32_t rows) -> uint32_t {
        if constexpr (SPLIT_MODE == SPLIT_ROWS) {
            return (rows * SPLIT_NUM) / SPLIT_DEN;
        } else {
            return 0;
        }
    };
    auto my_rows = [&](uint32_t rows) -> uint32_t {
        if constexpr (SPLIT_MODE == SPLIT_ROWS) {
            return rows - (rows * SPLIT_NUM) / SPLIT_DEN;
        } else if constexpr (SPLIT_MODE == SPLIT_ALL) {
            return 0;
        } else {
            return rows;
        }
    };

    // ---- the gather-zero boot, when it has NOT been moved to the reader --------------
    if constexpr (ZERO_R == 0) {
        if (is_root != 0) {
            MaybeDeviceZoneScope("writer_gather_zero");
            DataflowBuffer dfb(cb_partials_gathered);
            const uint32_t pages = dfb.get_total_size_bytes() / stat_bytes;
            for (uint32_t p = 0; p < pages; ++p) {
                const uint32_t base = p * stat_bytes;
                noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + face_bytes});
                noc.async_write_zeros(dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes});
            }
            noc.write_zeros_l1_barrier();
        }
    }

    const uint32_t root_x = mc.sender_x();
    const uint32_t root_y = mc.sender_y();
    // Does the reader own part of MY ship?  Never on the root when SPLIT_ROOT == 0.
    const bool split_here = (SPLIT_ROOT != 0) || (is_root == 0);
    const uint32_t my_faces = split_here ? MY_FACES : 0x3u;

    // ---- the ROOT's round ------------------------------------------------------------
    // (the mcast step is a callable so the SenderPipe object can live outside the loop --
    // see bench_reader.cpp on why the pipes are constructed exactly once per core.)
    auto root_round = [&](uint32_t blk, auto&& mcast_step) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        {
            MaybeDeviceZoneScope("writer_gather_ship");
            cb_reserve_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
            if constexpr (TOKENS) {
                // GO -- AFTER the reserve, so the reader cannot write into a ring compute
                // has not drained.
                if (split_here) {
                    cb_reserve_back(cb_tok_w2r, 1);
                    cb_push_back(cb_tok_w2r, 1);
                }
            }
            const uint32_t n = split_here ? my_rows(rows) : rows;
            const uint32_t a0 = split_here ? my_row0(rows) : 0u;
            if (n != 0) {
                ship(get_noc_addr(get_write_ptr(cb_partials_gathered)), r0 + a0, a0, n, my_faces);
            }
            noc_async_write_barrier();
            if constexpr (TOKENS) {
                if (split_here) {
                    // The reader's NoC0 half has LANDED.  Consumed even in the `tok` control
                    // (where the reader ships nothing), so that control measures the FULL
                    // round trip and (tok - base) is the edge's honest cost.
                    cb_wait_front(cb_tok_r2w, 1);
                    cb_pop_front(cb_tok_r2w, 1);
                }
            }
        }
        {
            MaybeDeviceZoneScope("writer_gather_wait");
            arrivals += GROUP_SIZE - 1;  // never a local self-signal: it is a non-atomic RMW
            sem1.wait_min(arrivals);
            if constexpr (ZERO_R != 0) {
                // The zeroing has to be visible HERE and only here -- the push is what
                // releases the ring to compute.  Waiting for it EARLIER (before the loop)
                // measured 0.920x on the focus geometry: it serialized the root's own ship
                // behind the reader's boot instead of overlapping them, which is the entire
                // point of moving the boot to NoC0.  It rides its OWN one-page FIFO, not the
                // per-round tok_r2w: sharing that FIFO would let this wait consume a
                // round-`landed` token (and vice versa), and since both are pure signals the
                // swap is SILENT -- the root would publish a ring whose own slot the reader
                // had not finished writing.
                if (blk == 0) {
                    cb_wait_front(cb_tok_zero, 1);
                    cb_pop_front(cb_tok_zero, 1);
                }
            }
            cb_push_back(cb_partials_gathered, GROUP_SIZE * BLOCK_ROWS);
        }
        mcast_step(rows);
    };

    // ---- a MEMBER's round ------------------------------------------------------------
    auto member_round = [&](uint32_t blk, auto&& mcast_step) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        if constexpr (NEED_ROUND_GATE) {
            if (blk != 0) {
                // The receive moved to the reader and took this kernel's only gate; take it
                // back before touching the root's ring again.
                MaybeDeviceZoneScope("writer_round_gate");
                cb_wait_front(cb_tok_r2w, 1);
                cb_pop_front(cb_tok_r2w, 1);
            }
        }
        {
            MaybeDeviceZoneScope("writer_gather_ship");
            if constexpr (TOKENS) {
                cb_reserve_back(cb_tok_w2r, 1);
                cb_push_back(cb_tok_w2r, 1);  // GO
            }
            const uint32_t n = my_rows(rows);
            if (n != 0) {
                ship(
                    get_noc_addr(root_x, root_y, get_write_ptr(cb_partials_gathered)),
                    r0 + my_row0(rows),
                    my_row0(rows),
                    n,
                    my_faces);
            }
            noc_async_write_barrier();  // data before signal
            if constexpr (TOKENS) {
                cb_wait_front(cb_tok_r2w, 1);  // BOTH halves have landed
                cb_pop_front(cb_tok_r2w, 1);
            }
            sem1.up(noc, root_x, root_y, 1);  // ONE inc, this kernel, always
        }
        mcast_step(rows);
    };

    if (is_root != 0) {
        if constexpr (MC_SEND_HERE) {
            auto sender = mc.sender(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                root_round(blk, [&](uint32_t rows) {
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
                });
            }
        } else {
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                root_round(blk, [](uint32_t) {});
            }
        }
    } else {
        if constexpr (MC_RECV_HERE) {
            auto receiver = mc.receiver(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                member_round(blk, [&](uint32_t rows) {
                    MaybeDeviceZoneScope("writer_mcast_recv");
                    cb_reserve_back(cb_row_final, rows);
                    receiver.receive();
                    cb_push_back(cb_row_final, rows);
                });
            }
        } else {
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                member_round(blk, [](uint32_t) {});
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
