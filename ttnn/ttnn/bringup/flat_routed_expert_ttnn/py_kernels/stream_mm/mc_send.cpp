// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast probe sender: multicasts TOTAL bytes from its own L1 in PIECE-byte writes to its rectangles (a ring of
// SLOTS pieces at the destination; nothing is consumed), then waits for all acks.
// CT: 0 TOTAL, 1 PIECE, 2 SLOTS, 3 POSTED (1: posted multicasts, no per-destination acks; 2: posted and no
//     multicast path reservation; 3: linked bursts: all pieces to one rectangle back to back, each linked to the next
//     except the last, rectangle after rectangle)
// RT: 0 source address, 1 ring address, 2 N_RECTS, then per rectangle start xy, end xy (this NoC's order), dests
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t total = get_compile_time_arg_val(0);
    constexpr uint32_t piece = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr bool posted = get_compile_time_arg_val(3) != 0;
    constexpr bool posted_reserve = get_compile_time_arg_val(3) != 2;  // POSTED 2: posted, no path reservation
    const uint32_t src = get_arg_val<uint32_t>(0), ring = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    uint64_t rect[4];
    uint32_t dests[4];
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t a0 = get_arg_val<uint32_t>(3 + 3 * r), a1 = get_arg_val<uint32_t>(4 + 3 * r);
        rect[r] = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        dests[r] = get_arg_val<uint32_t>(5 + 3 * r);
    }
    if constexpr (get_compile_time_arg_val(3) == 3) {
        constexpr uint32_t n_pieces = total / piece;
        for (uint32_t r = 0; r < n; ++r) {
            for (uint32_t i = 0; i < n_pieces; ++i) {
                const uint32_t off = (i % slots) * piece;
                noc_async_write_multicast(src + off, rect[r] | (ring + off), piece, dests[r], i + 1 < n_pieces);
            }
        }
        noc_async_write_barrier();
        return;
    }
    for (uint32_t i = 0; i < total / piece; ++i) {
        const uint32_t dst = ring + (i % slots) * piece;
        for (uint32_t r = 0; r < n; ++r) {
            if constexpr (posted) {
                ncrisc_noc_fast_write_any_len<noc_mode>(
                    noc_index,
                    write_cmd_buf,
                    src + (i % slots) * piece,
                    rect[r] | dst,
                    piece,
                    NOC_MULTICAST_WRITE_VC,
                    true,
                    false,
                    dests[r],
                    posted_reserve,
                    true);
            } else {
                noc_async_write_multicast(src + (i % slots) * piece, rect[r] | dst, piece, dests[r]);
            }
        }
    }
    noc_async_write_barrier();
    noc_async_posted_writes_flushed();
}
