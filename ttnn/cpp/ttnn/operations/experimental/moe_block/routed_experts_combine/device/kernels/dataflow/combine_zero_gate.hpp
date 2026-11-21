// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The combine's per-call zeroing inside its own program (the arena is allocated per call: its L1 holds another op's data
// between calls). Every core that owns combine state (a port's sender, an accumulator's reader; the flat expert's y
// writers for their words) zeroes it at its start and counts in on the chip's gate core (accumulator 0); once every one
// of them did, the gate releases every combine core of its chip. A kernel that writes into another core's state (an
// accumulator publishing ring layouts / returning credits to its ports, a reader granting writer slots, a sender's
// cross-chip barrier) waits for the release first. Remote chips are ordered by the senders' cross-chip barrier, which
// now also means "that chip is zeroed".
//
// The words live in the persistent `bar` tensor (16 words a core, every core, cumulative across calls):
//   bar[0] cross-chip barrier increments (port cores), bar[1] its calls,
//   bar[2] zeroed count-ins (gate core), bar[3] releases (every combine core),
//   bar[4 + slot] a kernel's own call count (slot: the RISC's, so two kernels of one core never share it).

#pragma once

#include <cstdint>
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

namespace rec {

constexpr uint32_t BAR_ZEROED = 2, BAR_RELEASED = 3, BAR_CALLS = 4;

// Starts zeroing [addr, addr + bytes) of this core's L1 (addr 16 B aligned). The bytes are zero once count_in()
// returns (its barrier); nothing may read or write them before that.
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t bytes) {
    Noc noc;
    noc.async_write_zeros(CoreLocalMem<uint32_t>(addr), bytes);
}

// count runs of `bytes` at addr, addr + stride, ...
FORCE_INLINE void zero_l1_strided(uint32_t addr, uint32_t bytes, uint32_t stride, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        zero_l1(addr + i * stride, bytes);
    }
}

// This kernel's call number (1, 2, ...).
FORCE_INLINE uint32_t next_call(uint32_t bar_addr, uint32_t slot) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bar_addr);
    const uint32_t c = w[BAR_CALLS + slot] + 1;
    w[BAR_CALLS + slot] = c;
    return c;
}

// After this kernel's zeroing (zero_l1 / zero_l1_strided issued): wait for it, then count in on the gate.
FORCE_INLINE void count_in(uint32_t gate_x, uint32_t gate_y, uint32_t bar_addr) {
    Noc noc;
    noc.write_zeros_l1_barrier();
    noc_semaphore_inc(get_noc_addr(gate_x, gate_y, bar_addr + 4 * BAR_ZEROED), 1);
    noc_async_atomic_barrier();
}

// The gate (accumulator 0's reader): every one of the chip's nz count-ins of call `call` arrived -> release each
// combine core (n_cores (x, y) pairs from runtime arg `a`).
FORCE_INLINE void release(uint32_t bar_addr, uint32_t call, uint32_t nz, uint32_t n_cores, size_t a) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bar_addr);
    const uint32_t tgt = call * nz;
    do {
        invalidate_l1_cache();
    } while (static_cast<int32_t>(w[BAR_ZEROED] - tgt) < 0);
    for (uint32_t i = 0; i < n_cores; ++i, a += 2) {
        noc_semaphore_inc(
            get_noc_addr(moe_block_args::arg(a), moe_block_args::arg(a + 1), bar_addr + 4 * BAR_RELEASED), 1);
    }
    noc_async_atomic_barrier();
}

// Every core of this chip zeroed its combine state for call `call`.
FORCE_INLINE void wait_released(uint32_t bar_addr, uint32_t call) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bar_addr);
    do {
        invalidate_l1_cache();
    } while (static_cast<int32_t>(w[BAR_RELEASED] - call) < 0);
}

}  // namespace rec
