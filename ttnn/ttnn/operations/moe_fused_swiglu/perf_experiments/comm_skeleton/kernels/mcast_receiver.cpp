// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// comm_skeleton probe 4: THE GRID-WIDE MULTICAST RENDEZVOUS — RECEIVER HALF.
//
// Runs on the READER RISC-V (NOC0) of EVERY core in the worker rect, which is where the op's h
// receive loop lives. Per round a receiver does exactly two things and no payload work:
//
//   * ACK the round's root ("my slot is free") — this is sent BEFORE waiting for the data, which is
//     what makes the sender/receiver split deadlock-free in the real op.
//   * WAIT for the round's ready signal.
//
// DEPTH is the rolling-window knob (the op's `DEPTH_H`): a receiver pre-acks DEPTH rounds up front
// and then acks round r+DEPTH as it clears round r. DEPTH=1 is the fully serialised chain — every
// round's ack blocked behind the previous round's data — and larger DEPTH is the pipelined form.
// Sweeping it prices the serialisation directly.
//
// A ROOT does not receive its own broadcast (multicast excludes the source), so it expects one
// fewer arrival on the rounds it sends; every non-root core in that column DOES receive it. The
// skeleton counts arrivals on ONE monotone semaphore rather than the op's per-slot ready flags —
// the primitive COUNT and the pipelining depth are identical, and no payload is being told apart.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t HGROUPS = get_compile_time_arg_val(0);
constexpr uint32_t DEPTH = get_compile_time_arg_val(1);
constexpr uint32_t STAGE = get_compile_time_arg_val(2);
constexpr uint32_t SEM_FREE = get_compile_time_arg_val(3);
constexpr uint32_t SEM_RDY = get_compile_time_arg_val(4);
constexpr uint32_t VERIFY = get_compile_time_arg_val(5);
constexpr uint32_t CB_MCAST = get_compile_time_arg_val(6);
constexpr uint32_t TA_BASE = 7;
constexpr auto out_args = TensorAccessorArgs<TA_BASE>();

// STAGE: 0 = ack only, 1 = ack+data, 2 = full, 3 = ack+ready with NO data (see mcast_sender.cpp).
// The wait is what SERIALISES the rounds, so it is present exactly when the sender signals ready.
constexpr bool DO_READY = (STAGE == 2) || (STAGE == 3);

constexpr uint32_t RT_ROOTS = 7;  // runtime-arg index where the per-column root coordinates start

void kernel_main() {
    const uint32_t my_col = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_page = get_arg_val<uint32_t>(3);
    const uint32_t my_core_idx = get_arg_val<uint32_t>(4);
    const uint32_t ROUNDS = get_arg_val<uint32_t>(5);
    const uint32_t PAYLOAD_BYTES = get_arg_val<uint32_t>(6);

    const uint32_t free_sem_addr = static_cast<uint32_t>(get_semaphore(SEM_FREE));
    auto* rdy_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_RDY)));

    // ack(r) — one atomic increment at the root of round r's column.
    auto ack = [&](uint32_t r) {
        const uint32_t c = r % HGROUPS;
        const uint32_t rvx = get_arg_val<uint32_t>(RT_ROOTS + 2 * c + 0);
        const uint32_t rvy = get_arg_val<uint32_t>(RT_ROOTS + 2 * c + 1);
        noc_semaphore_inc(get_noc_addr(rvx, rvy, free_sem_addr), 1);
        noc_async_atomic_barrier();
    };

    for (uint32_t r = 0; r < DEPTH && r < ROUNDS; ++r) {
        ack(r);
    }

    uint32_t rdy_expected = 0;
    for (uint32_t r = 0; r < ROUNDS; ++r) {
        if constexpr (DO_READY) {
            if (!(is_root && (r % HGROUPS) == my_col)) {
                ++rdy_expected;
                noc_semaphore_wait_min(rdy_sem, rdy_expected);
            }
        }
        const uint32_t nxt = r + DEPTH;
        if (nxt < ROUNDS) {
            ack(nxt);
        }
    }

    // CORRECTNESS GATE (STAGE >= 1 only). Every sender broadcast the SAME bytes, so after the last
    // round every core's slot must hold them. Publishing the slot lets the host prove the multicast
    // actually LANDED rather than trusting that a fast run did the work. One write per core, after
    // the measured loop — it lands in the intercept, not the per-round slope.
    if constexpr (VERIFY) {
        const auto out_acc = TensorAccessor(out_args, out_addr, out_page);
        noc_async_write(get_read_ptr(CB_MCAST), out_acc.get_noc_addr(my_core_idx), PAYLOAD_BYTES);
        noc_async_write_barrier();
    }
}
