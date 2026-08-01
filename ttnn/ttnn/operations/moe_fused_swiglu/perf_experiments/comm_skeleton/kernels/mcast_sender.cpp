// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// comm_skeleton probe 4: THE GRID-WIDE MULTICAST RENDEZVOUS — SENDER HALF.
//
// This is the op's phase-2 h-broadcast round with the payload removed: one root per grid column,
// round r broadcast by the root of column (r % HGROUPS), the whole worker rect receiving. Runs on
// the WRITER RISC-V (NOC1), which is where the real op puts it (`HSEND_WRITER`).
//
// Per round the sender pays three separable things, peeled by STAGE so each can be priced alone:
//   STAGE 0  the ACK INCAST only     — wait for NUM_CORES receivers to signal their slot is free.
//   STAGE 1  + the DATA MULTICAST    — one `noc_async_write_multicast` of PAYLOAD_BYTES + barrier.
//   STAGE 2  + the READY SIGNAL      — `noc_semaphore_inc_multicast` + atomic barrier (the full round).
//   STAGE 3  ack + READY, NO DATA    — the same fully-serialised round with only the payload write
//                                      removed. STAGE 2 minus STAGE 3 is the data multicast priced
//                                      under IDENTICAL ordering; STAGE 1 cannot do that job, because
//                                      without the ready signal nothing serialises the rounds and the
//                                      measurement becomes a race between sender and receivers.
//
// The rect is given in NOC1 ROUTING ORDER (start = far corner) because the multicast hardware walks
// from `start` in the NoC's own direction, which is the reverse of NOC0's — the same convention the
// real writer's rect uses.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t HGROUPS = get_compile_time_arg_val(0);
constexpr uint32_t NUM_CORES = get_compile_time_arg_val(1);
constexpr uint32_t STAGE = get_compile_time_arg_val(2);
constexpr uint32_t SEM_FREE = get_compile_time_arg_val(3);
constexpr uint32_t SEM_RDY = get_compile_time_arg_val(4);
constexpr uint32_t CB_MCAST = get_compile_time_arg_val(5);
constexpr uint32_t TA_BASE = 6;
constexpr auto in_args = TensorAccessorArgs<TA_BASE>();

constexpr bool DO_DATA = (STAGE == 1) || (STAGE == 2);
constexpr bool DO_READY = (STAGE == 2) || (STAGE == 3);

void kernel_main() {
    const uint32_t my_col = get_arg_val<uint32_t>(0);
    const uint32_t rect_x0 = get_arg_val<uint32_t>(1);  // NOC1 order: far corner first
    const uint32_t rect_y0 = get_arg_val<uint32_t>(2);
    const uint32_t rect_x1 = get_arg_val<uint32_t>(3);
    const uint32_t rect_y1 = get_arg_val<uint32_t>(4);
    const uint32_t in_addr = get_arg_val<uint32_t>(5);
    const uint32_t in_page = get_arg_val<uint32_t>(6);
    const uint32_t ROUNDS = get_arg_val<uint32_t>(7);
    const uint32_t PAYLOAD_BYTES = get_arg_val<uint32_t>(8);

    cb_reserve_back(CB_MCAST, 1);
    const uint32_t src_l1 = get_write_ptr(CB_MCAST);

    // Every sender loads the SAME source page, so after any number of rounds every core's slot must
    // hold identical, KNOWN bytes — that is what the receiver's publish-and-compare gate checks.
    // One read, outside the measured loop.
    {
        const auto in_acc = TensorAccessor(in_args, in_addr, in_page);
        noc_async_read(in_acc.get_noc_addr(0), src_l1, PAYLOAD_BYTES);
        noc_async_read_barrier();
    }

    // Same CB index, same size, same declaration order on every core, so the landing address is the
    // sender's own write pointer. (The real op derives its slot the same way, for the same reason.)
    const uint64_t data_dst = get_noc_multicast_addr(rect_x0, rect_y0, rect_x1, rect_y1, src_l1);
    const uint64_t rdy_dst =
        get_noc_multicast_addr(rect_x0, rect_y0, rect_x1, rect_y1, static_cast<uint32_t>(get_semaphore(SEM_RDY)));

    auto* free_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_FREE)));

    uint32_t free_expected = 0;
    for (uint32_t r = my_col; r < ROUNDS; r += HGROUPS) {
        free_expected += NUM_CORES;
        noc_semaphore_wait_min(free_sem, free_expected);
        if constexpr (DO_DATA) {
            noc_async_write_multicast(src_l1, data_dst, PAYLOAD_BYTES, NUM_CORES - 1, /*linked=*/false);
            noc_async_write_barrier();
        }
        if constexpr (DO_READY) {
            noc_semaphore_inc_multicast(rdy_dst, 1, NUM_CORES - 1);
            noc_async_atomic_barrier();
        }
    }
}
