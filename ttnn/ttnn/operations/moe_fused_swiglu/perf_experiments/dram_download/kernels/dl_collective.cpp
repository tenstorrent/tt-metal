// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DATA-MOVEMENT FLOOR — the op's three COLLECTIVES with no compute and no round rendezvous.
//
// Runs on both data-movement RISCs (private CB sets, see dl_stream.cpp for why). Each phase moves the
// op's real byte volume to the op's real destinations, then closes with one barrier:
//
//   P_XMCAST   the x row-multicast. Column c's injector sends its staged tile-row to the OTHER
//              HGROUPS-1 columns of its grid row. m_eff rounds. 19.50 MB of L1 writes per M-block.
//   P_REDUCE   the column reduce-scatter. Every core unicasts its 1/KGROUPS slice of gate AND up into
//              each of the KGROUPS slice owners down its column, then the owner ships the finished
//              slice to the column root. 9.19 MB.
//   P_HGATHER  the h all-gather. Each of the HGROUPS column roots multicasts its m_eff*HN_PAD bfp8 h
//              block to the WHOLE grid. 50.55 MB — the largest single traffic term in the op.
//
// WHAT IS DELIBERATELY ABSENT: the per-round semaphore rendezvous. The op's h rounds are strictly
// ordered (column r sends in round r, and round r+1's sender waits on every receiver clearing r), and
// that ordering is measured at ~34 us per M-block of pure rendezvous. Here every sender fires as soon
// as it can, so this is the BANDWIDTH floor of the same traffic — the gap between it and the op is the
// rendezvous, which is the number worth having. Nothing is consumed, so no flow control is needed and
// receivers may be overwritten; that is sound for a timing measurement and unsound for anything else.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t P_XMCAST = get_compile_time_arg_val(0);
constexpr uint32_t P_REDUCE = get_compile_time_arg_val(1);
constexpr uint32_t P_HGATHER = get_compile_time_arg_val(2);
constexpr uint32_t HGROUPS = get_compile_time_arg_val(3);
constexpr uint32_t KGROUPS = get_compile_time_arg_val(4);
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(5);
constexpr uint32_t NUM_CORES = get_compile_time_arg_val(6);
constexpr uint32_t CB_SRC = get_compile_time_arg_val(7);   // my payload source (never read back)
constexpr uint32_t CB_LAND = get_compile_time_arg_val(8);  // where OTHER cores' payloads land
constexpr uint32_t POSTED = get_compile_time_arg_val(9);   // 1 = posted multicasts (no write-acks)

// POSTED MULTICAST. `noc_async_write_multicast` is NON-posted: every destination returns a write-ack
// and the sender's `noc_async_write_barrier` waits for all of them, so a whole-grid broadcast pays an
// (NUM_CORES-1)-way ack incast per round. A posted write asks for no ack at all — the low-level
// `ncrisc_noc_fast_write_any_len` exposes it as a `posted` flag that the public wrapper does not
// plumb through. There is then nothing to wait for, so the round closes with
// `noc_async_writes_flushed()` (SENT) instead of `noc_async_write_barrier()` (LANDED).
//
// THIS IS MEASUREMENT-ONLY AND NOT CORRECT FOR THE REAL OP. Posted writes give the receiver no landing
// guarantee whatsoever, and "flushed" means the data left this core, not that it arrived. Any real use
// would need a landing guarantee that posted writes cannot provide — the same data-before-signal
// hazard that made the op's `HSIG=Counter` path hang. Here nothing is consumed, so it is safe.
FORCE_INLINE void mcast(uint32_t src_addr, uint64_t dst, uint32_t size, uint32_t num_dests) {
    if constexpr (POSTED) {
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf));
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc_index,
            write_cmd_buf,
            src_addr,
            dst,
            size,
            NOC_MULTICAST_WRITE_VC,
            true /* mcast */,
            false /* linked */,
            num_dests,
            true /* multicast_path_reserve */,
            true /* POSTED */);
    } else {
        noc_async_write_multicast(src_addr, dst, size, num_dests, /*linked=*/false);
    }
}

FORCE_INLINE void mcast_close() {
    if constexpr (POSTED) {
        noc_async_writes_flushed();  // SENT, not landed — nothing acks a posted write
    } else {
        noc_async_write_barrier();
    }
}

void kernel_main() {
    uint32_t i = 0;
    const uint32_t my_col = get_arg_val<uint32_t>(i++);
    const uint32_t my_row = get_arg_val<uint32_t>(i++);
    const uint32_t m_eff = get_arg_val<uint32_t>(i++);
    const uint32_t hn = get_arg_val<uint32_t>(i++);
    const uint32_t kr = get_arg_val<uint32_t>(i++);
    const uint32_t x_rows = get_arg_val<uint32_t>(i++);   // tile-rows I inject (0 = not an injector)
    const uint32_t is_root = get_arg_val<uint32_t>(i++);  // am I my column's reduce root / h sender
    // My grid row as a multicast rectangle (for x), and the whole grid (for h).
    const uint32_t row_x0 = get_arg_val<uint32_t>(i++);
    const uint32_t row_y0 = get_arg_val<uint32_t>(i++);
    const uint32_t row_x1 = get_arg_val<uint32_t>(i++);
    const uint32_t row_y1 = get_arg_val<uint32_t>(i++);
    const uint32_t all_x0 = get_arg_val<uint32_t>(i++);
    const uint32_t all_y0 = get_arg_val<uint32_t>(i++);
    const uint32_t all_x1 = get_arg_val<uint32_t>(i++);
    const uint32_t all_y1 = get_arg_val<uint32_t>(i++);
    // Physical NoC coords of my column's cores, for the reduce-scatter unicasts.
    const uint32_t col_base = i;  // KGROUPS pairs of (x, y) follow

    const uint32_t src = get_write_ptr(CB_SRC);
    const uint32_t land = get_write_ptr(CB_LAND);

    // ---- P_XMCAST: my staged x tile-rows out to the rest of my grid row ----
    //
    // SOURCE ADDRESS == DESTINATION ADDRESS, and that is load-bearing. The sender sits INSIDE its own
    // row rectangle, so a send with `src_l1 != dst_l1` is a LOOPBACK multicast (mcast_pipe.inl:
    // `loopback = in_rect_ && src_l1 != dst_l1`) and needs the loopback API with a fan-out of HGROUPS.
    // Calling the plain API with the exclude-source count HGROUPS-1 against a mismatched rectangle
    // hangs the grid — which it did. Sending from `land` makes the send genuinely exclude-source, the
    // same reason the real reader lands x with a self-copy instead of a loopback multicast.
    if constexpr (P_XMCAST) {
        if (x_rows) {
            const uint32_t xbytes = kr * BFP8_TILE;
            for (uint32_t r = 0; r < x_rows; ++r) {
                const uint64_t dst = get_noc_multicast_addr(row_x0, row_y0, row_x1, row_y1, land + r * xbytes);
                mcast(land + r * xbytes, dst, xbytes, HGROUPS - 1);
            }
            mcast_close();
        }
    }

    // ---- P_REDUCE: column all-to-all slice scatter, then owner -> root gather ----
    if constexpr (P_REDUCE) {
        const uint32_t block = m_eff * hn * BFP8_TILE;
        const uint32_t slice = block / KGROUPS;  // my 1/KGROUPS share of the block
        // Scatter: send slice `o` of BOTH gate and up to owner `o`, for every owner in my column.
        for (uint32_t o = 0; o < KGROUPS; ++o) {
            const uint32_t ox = get_arg_val<uint32_t>(col_base + 2 * o);
            const uint32_t oy = get_arg_val<uint32_t>(col_base + 2 * o + 1);
            const uint64_t dst = get_noc_addr(ox, oy, land + my_row * slice);
            noc_async_write(src + o * slice, dst, slice);                    // gate
            noc_async_write(src + o * slice, dst + KGROUPS * slice, slice);  // up
        }
        noc_async_write_barrier();
        // Gather: every owner ships its finished gate+up slice to the column root (row 0).
        {
            const uint32_t rx = get_arg_val<uint32_t>(col_base + 0);
            const uint32_t ry = get_arg_val<uint32_t>(col_base + 1);
            const uint64_t dst = get_noc_addr(rx, ry, land + my_row * 2 * slice);
            noc_async_write(src, dst, 2 * slice);
            noc_async_write_barrier();
        }
    }

    // ---- P_HGATHER: each column root multicasts its h block to the whole grid ----
    if constexpr (P_HGATHER) {
        if (is_root) {
            const uint32_t hbytes = m_eff * hn * BFP8_TILE;
            const uint64_t dst = get_noc_multicast_addr(all_x0, all_y0, all_x1, all_y1, land);
            // EXCLUDE-source, exactly like the op — and `src_l1 == dst_l1` is what makes it so (see
            // the P_XMCAST note): the sender is inside the grid rectangle.
            mcast(land, dst, hbytes, NUM_CORES - 1);
            mcast_close();
        }
    }
}
