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
    // Phase 3 — receiver drains to DRAM (Approach #2). When DRAM_DRAIN is set, a consumed slot is
    // written from L1 to this chip's DRAM output buffer (page dram_base_page + consumed) over NOC_0
    // before its credit is returned, so the credit provably means "the token is in DRAM". The producer
    // writes L1->eth over NOC_1, so the two directions do not contend on the same NoC.
    constexpr uint32_t variant = get_compile_time_arg_val(8);
    constexpr uint32_t dram_base_page = get_compile_time_arg_val(9);
    constexpr uint32_t dram_bank_base_addr = get_compile_time_arg_val(10);
    constexpr auto dram_out_args = TensorAccessorArgs<11>();
    constexpr bool DRAM_DRAIN = (variant & 64u) != 0;  // Approach #2

    volatile tt_l1_ptr uint32_t* data_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_addr);
    // Bumped via a NoC atomic to our OWN core: the proven idiom for cross-RISC visibility on the same
    // core (a plain store can sit in a write buffer where the other RISC will not see it).
    const uint64_t my_credits_noc = get_noc_addr(my_noc_x, my_noc_y, credits_to_return_addr);
    const auto dram_out = TensorAccessor(dram_out_args, dram_bank_base_addr);

    uint32_t consumed = 0;
    {
        DeviceZoneScopedN("RECEIVER_LOOP");
        while (consumed < num_tokens) {
            invalidate_l1_cache();
            const uint32_t arrived = *data_ready;
            while (consumed < arrived) {
                DeviceZoneScopedN("RECEIVER_RECV");
                if constexpr (DRAM_DRAIN) {
                    // Drain the landed L1 slot to its DRAM page, then wait for it to actually land
                    // before returning the credit. The barrier is what makes the credit a real
                    // "in DRAM" guarantee — and what lets this experiment reveal whether the DRAM
                    // write, not the fabric, is the bottleneck.
                    const uint32_t slot = consumed % num_slots;
                    noc_async_write(
                        recv_buf_addr + slot * chunk_size_bytes,
                        dram_out.get_noc_addr(dram_base_page + consumed),
                        chunk_size_bytes);
                    noc_async_write_barrier();
                } else {
                    // NOP ACK (baseline L1-only path).
                    (void)recv_buf_addr;
                    (void)chunk_size_bytes;
                    (void)num_slots;
                }
                consumed++;
                // Hand the credit to the producer on this core; it forwards it over fabric.
                noc_semaphore_inc(my_credits_noc, 1);
            }
        }
    }

    noc_async_writes_flushed();
}
