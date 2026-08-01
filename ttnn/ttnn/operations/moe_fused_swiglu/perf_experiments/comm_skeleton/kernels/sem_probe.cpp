// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// comm_skeleton probe 3: SEMAPHORE COST.
//
// Five roles out of one source. Each is a bare loop over ONE semaphore primitive so the slope of
// ns vs N_OPS is that primitive's cost with nothing else in the way.
//
//   INC_P2P        N_OPS `noc_semaphore_inc` at one peer core. BARRIER_EACH selects the op's own
//                  spelling (inc + `noc_async_atomic_barrier()` every time, the form used in the
//                  writer's reduce-child edge and h-slice send) versus the batched form (N incs,
//                  one barrier). The gap is the cost of the per-inc barrier.
//   INCAST_SENDER  the same loop, but every core in the grid aims at ONE root. Paired with
//                  INCAST_ROOT this is exactly the op's per-round ack incast (`SEM_H_FREE`), with
//                  the payload and everything downstream of it removed.
//   INCAST_ROOT    N_OPS successive `noc_semaphore_wait_min` on the incast target, each raised by
//                  N_SENDERS. Its duration is how long the root's semaphore takes to ABSORB
//                  N_OPS*N_SENDERS remote atomics — the incast throughput the op's h all-gather
//                  pays once per round.
//   WAIT_SAT       N_OPS `noc_semaphore_wait_min` on an ALREADY-SATISFIED semaphore (initial value
//                  set high host-side). Pure call overhead: the floor under every wait in the op.
//   MCAST_SEM      N_OPS `noc_semaphore_inc_multicast` over the whole worker rect. The op's
//                  ready-signal, alone.
//
// Nothing here moves a payload byte.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

#define ROLE_INC_P2P 0
#define ROLE_INCAST_SENDER 1
#define ROLE_INCAST_ROOT 2
#define ROLE_WAIT_SAT 3
#define ROLE_MCAST_SEM 4

constexpr uint32_t ROLE = get_compile_time_arg_val(0);
constexpr uint32_t BARRIER_EACH = get_compile_time_arg_val(1);
constexpr uint32_t SEM_TARGET = get_compile_time_arg_val(2);  // the semaphore incremented / waited on
constexpr uint32_t SEM_SAT = get_compile_time_arg_val(3);     // pre-satisfied semaphore (WAIT_SAT)
constexpr uint32_t N_DESTS = get_compile_time_arg_val(4);     // multicast fan-out (excludes self)

void kernel_main() {
    const uint32_t peer_vx = get_arg_val<uint32_t>(0);
    const uint32_t peer_vy = get_arg_val<uint32_t>(1);
    const uint32_t rect_x0 = get_arg_val<uint32_t>(2);
    const uint32_t rect_y0 = get_arg_val<uint32_t>(3);
    const uint32_t rect_x1 = get_arg_val<uint32_t>(4);
    const uint32_t rect_y1 = get_arg_val<uint32_t>(5);
    // RUNTIME (see the note in cb_probe.cpp): one build serves the whole sweep, and the op's own
    // round counts are runtime values too.
    const uint32_t N_OPS = get_arg_val<uint32_t>(6);
    const uint32_t N_SENDERS = get_arg_val<uint32_t>(7);  // incast fan-in

    const uint32_t sem_target_addr = static_cast<uint32_t>(get_semaphore(SEM_TARGET));

    if constexpr (ROLE == ROLE_INC_P2P || ROLE == ROLE_INCAST_SENDER) {
        const uint64_t dst = get_noc_addr(peer_vx, peer_vy, sem_target_addr);
        for (uint32_t i = 0; i < N_OPS; ++i) {
            noc_semaphore_inc(dst, 1);
            if constexpr (BARRIER_EACH) {
                noc_async_atomic_barrier();
            }
        }
        if constexpr (!BARRIER_EACH) {
            noc_async_atomic_barrier();
        }
    } else if constexpr (ROLE == ROLE_INCAST_ROOT) {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_target_addr);
        uint32_t expected = 0;
        for (uint32_t i = 0; i < N_OPS; ++i) {
            expected += N_SENDERS;
            noc_semaphore_wait_min(sem, expected);
        }
    } else if constexpr (ROLE == ROLE_WAIT_SAT) {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_SAT)));
        for (uint32_t i = 0; i < N_OPS; ++i) {
            noc_semaphore_wait_min(sem, 1);
        }
    } else {  // ROLE_MCAST_SEM
        const uint64_t rect = get_noc_multicast_addr(rect_x0, rect_y0, rect_x1, rect_y1, sem_target_addr);
        for (uint32_t i = 0; i < N_OPS; ++i) {
            noc_semaphore_inc_multicast(rect, 1, N_DESTS);
            if constexpr (BARRIER_EACH) {
                noc_async_atomic_barrier();
            }
        }
        if constexpr (!BARRIER_EACH) {
            noc_async_atomic_barrier();
        }
    }
}
