// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver kernel (reader RISC). Shares its core with the producer kernel.
//
// It holds NO fabric connection, and needs none: being a fabric destination requires no handshake or
// registration at all — the peer's eth RISC writes straight into the address its packet header names
// and bumps `data_ready` with the fused atomic-inc. So the receiver's whole inbound path is polling a
// word in its own L1. (That also means its L1 is unprotected, which is exactly why the credit loop
// below has to exist.)
//
// Consuming a token is a NOP for now. The credit for it is handed to the co-located producer, which
// owns the eth channel's single connection and forwards it over fabric.
//
// `data_ready` is written only by the remote eth RISC and `credits_to_return` only by us, so both are
// single-writer monotonic counters: we keep our own `consumed` count and work on the difference. No
// read-modify-write, so nothing to race with the producer reading `credits_to_return`.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t num_tokens = get_compile_time_arg_val(0);
    constexpr uint32_t num_slots = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t recv_buf_addr = get_compile_time_arg_val(3);
    constexpr uint32_t data_ready_addr = get_compile_time_arg_val(4);
    constexpr uint32_t credits_to_return_addr = get_compile_time_arg_val(5);
    constexpr uint32_t my_noc_x = get_compile_time_arg_val(6);
    constexpr uint32_t my_noc_y = get_compile_time_arg_val(7);

    volatile tt_l1_ptr uint32_t* data_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_addr);
    // Bumped via a NoC atomic to our OWN core: the proven idiom for cross-RISC visibility on the same
    // core (a plain store can sit in a write buffer where the other RISC will not see it).
    const uint64_t my_credits_noc = get_noc_addr(my_noc_x, my_noc_y, credits_to_return_addr);

    uint32_t consumed = 0;
    {
        DeviceZoneScopedN("RECEIVER_LOOP");
        while (consumed < num_tokens) {
            invalidate_l1_cache();
            const uint32_t arrived = *data_ready;
            while (consumed < arrived) {
                DeviceZoneScopedN("RECEIVER_RECV");
                // "Process" slot `consumed % num_slots` — a NOP ACK for now; Phase 3 drains to DRAM.
                (void)recv_buf_addr;
                (void)chunk_size_bytes;
                (void)num_slots;
                consumed++;
                // Hand the credit to the producer on this core; it forwards it over fabric.
                noc_semaphore_inc(my_credits_noc, 1);
            }
        }
    }

    noc_async_writes_flushed();
}
