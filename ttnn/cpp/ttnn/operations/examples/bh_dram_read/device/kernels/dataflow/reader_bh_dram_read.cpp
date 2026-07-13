// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Reader-only kernel. One instance runs per DRAM bank. It streams the bytes of
// the input tensor that reside in this core's assigned bank into L1 and
// discards them (no compute, no writer).
//
// Peak-bandwidth pattern (copied from the MoE dm0.cpp kernel): each core reads
// its bank's contiguous region in maximum-size NOC packets (NOC_MAX_BURST_SIZE,
// 16 KB on Blackhole), keeping NUM_TRIDS transactions in flight via NOC
// transaction ids. set_state is issued once (the bank/size never change for
// this core), then with_state reads are fired back-to-back; a per-trid barrier
// is taken only once the trid ring is full.

// Outstanding transactions (buffering depth). Set by the program factory; the
// L1 scratch (CB) holds NUM_TRIDS packets. Valid trids are 1..15 on Blackhole.
constexpr uint32_t NUM_TRIDS = get_compile_time_arg_val(0);
#define ADVANCE_TRID(t)        \
    do {                       \
        (t)++;                 \
        if ((t) > NUM_TRIDS) { \
            (t) = 1;           \
        }                      \
    } while (0)

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);      // buffer base address (in-bank byte offset base)
    const uint32_t region_bytes = get_arg_val<uint32_t>(1);  // bytes this core reads from its bank
    const uint32_t bank_id = get_arg_val<uint32_t>(2);       // this core's DRAM bank

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t PACKET = NOC_MAX_BURST_SIZE;  // 16 KB on Blackhole (== several tiles)

    // L1 scratch base. The CB is sized (in the program factory) to hold
    // NUM_TRIDS packets; slot k is reused only after its trid has been
    // barriered, so we just index into it directly (no reserve/push/pop).
    const uint32_t l1_base = get_write_ptr(cb_id_in0);

    const uint64_t bank_noc_base = get_noc_addr_from_bank_id<true>(bank_id, 0);

    // Always read full max-size packets and round the count up: the final
    // packet may over-read past region_bytes, which is fine since every byte is
    // discarded. This keeps the hot loop a single constant-size transaction.
    const uint32_t num_packets = (region_bytes + PACKET - 1) / PACKET;

    {
        // Captured by the device profiler as zone "DRAM_READ".
        DeviceZoneScopedN("DRAM_READ");

        // Stream constant max-size packets with NUM_TRIDS in flight.
        noc_async_read_one_packet_set_state(bank_noc_base, PACKET);
        uint32_t trid_issue = 1, trid_wait = 1, in_flight = 0;
        uint32_t offset = src_addr;
        for (uint32_t i = 0; i < num_packets; ++i) {
            noc_async_read_set_trid(trid_issue);
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/true>(
                bank_noc_base, offset, l1_base + (trid_issue - 1) * PACKET, trid_issue);
            offset += PACKET;
            ADVANCE_TRID(trid_issue);
            if (++in_flight == NUM_TRIDS) {
                noc_async_read_barrier_with_trid(trid_wait);
                ADVANCE_TRID(trid_wait);
                --in_flight;
            }
        }
        // Drain the ring.
        while (in_flight > 0) {
            noc_async_read_barrier_with_trid(trid_wait);
            ADVANCE_TRID(trid_wait);
            --in_flight;
        }
    }
}
