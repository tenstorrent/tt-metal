// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Chained M-split big-M h relay (BRISC, NOC0) on a core of its own. For each M-group g and virtual expert v, once all
// the group's cores have written their slices into the group's buffer here (GATH_g), writes the assembled h into
// every chain head of the group (unicast, the head's h_all) as soon as that head has freed its h_all (the head's
// free word here, written by se5_recv.cpp), then bumps the head's HARR. No multicast.
// The group buffer is reused for v + 1 only after every core finished down(v) (the cores' "go" gating), which implies
// every head received h(v).
//
// CT: 0 NUM_V, 1 GROUP_NCC, 2 HALF_BYTES, 3 HARR_SEM, 4 G, 5 HEADS (per group), 6 H_PIECES (HARR counts pieces)
// RT: 0 h_all address (compute cores), 1 own buffer base, then per group: GATH sem id, then per head: head xy, sem id
//     of the head's free word here
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_v = get_compile_time_arg_val(0);
    constexpr uint32_t group_ncc = get_compile_time_arg_val(1);
    constexpr uint32_t half_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t groups = get_compile_time_arg_val(4);
    constexpr uint32_t heads = get_compile_time_arg_val(5);
    constexpr uint32_t pieces = get_compile_time_arg_val(6);
    constexpr uint32_t piece_bytes = half_bytes / pieces;

    const uint32_t h_all = get_arg_val<uint32_t>(0);
    const uint32_t own = get_arg_val<uint32_t>(1);
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    volatile tt_l1_ptr uint32_t* gath[groups];
    uint64_t head_hall[groups][heads], head_harr[groups][heads];
    volatile tt_l1_ptr uint32_t* head_free[groups][heads];
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t a = 2 + g * (1 + 2 * heads);
        gath[g] = sem(get_arg_val<uint32_t>(a));
        for (uint32_t k = 0; k < heads; ++k) {
            const uint32_t xy = get_arg_val<uint32_t>(a + 1 + 2 * k);
            head_hall[g][k] = get_noc_addr(xy >> 16, xy & 0xFFFF, h_all);
            head_harr[g][k] = get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(harr_sem_id));
            head_free[g][k] = sem(get_arg_val<uint32_t>(a + 2 + 2 * k));
        }
    }

    uint32_t hv[groups], sent[groups], pc[groups];  // virtual expert being sent per group, heads done, next piece
    for (uint32_t g = 0; g < groups; ++g) {
        hv[g] = 0;
        sent[g] = 0;
        pc[g] = 0;
    }
    uint32_t left = groups * num_v;
    while (left) {
        invalidate_l1_cache();
        for (uint32_t g = 0; g < groups; ++g) {
            if (hv[g] == num_v || *gath[g] < group_ncc * (hv[g] + 1)) {
                continue;
            }
            // One piece per pass per group: write, wait for the acks, bump the head's piece counter.
            const uint32_t k = sent[g];
            if (*head_free[g][k] >= hv[g]) {
                const uint32_t off = pc[g] * piece_bytes;
                noc_async_write(own + g * half_bytes + off, head_hall[g][k] + off, piece_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(head_harr[g][k], 1);
                if (++pc[g] == pieces) {
                    pc[g] = 0;
                    if (++sent[g] == heads) {
                        sent[g] = 0;
                        ++hv[g];
                        --left;
                    }
                }
            }
        }
    }
    noc_async_atomic_barrier();
}
