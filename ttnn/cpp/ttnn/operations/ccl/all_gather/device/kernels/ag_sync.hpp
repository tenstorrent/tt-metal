// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// all_gather synchronisation helpers and the address rule.
//
// Address rule. Two frames, never mixed:
//   - Fabric frame: every address placed in a fabric command header. The header setters (to_noc_unicast_write,
//     to_noc_unicast_scatter_write, the atomic-inc setters) re-translate it for the NOC the receiving router writes
//     with (edm_to_local_chip_noc).
//       - Record (data) addresses come from FabricFrameAddr, i.e. addrgen_detail::get_noc_address, the frame the
//         fabric write setters expect.
//       - Semaphore and mirror addresses come from fabric_frame_addr(x, y, l1) = safe_get_noc_addr(x, y, l1, 0).
//         Never get_noc_addr(x, y, l1) on the NOC1 writer: that is a NOC1-frame address.
//   - Own frame: every read or write a RISC issues itself (accessor addresses on its own NOC, DstRun::local). The
//     fabric frame is not usable there: on BH a DRAM record address names the bank's NOC1 endpoint, which a NOC0
//     read or write does not reach. So the reader re-reads relays from, and writes local copies to, the own-NOC
//     segment addresses PacketBuilder::next reports, never record addresses.
//   - Local semaphore operations use the RISC's own NOC (get_noc_addr(l1)).
//
// Sync protocol (counted units, subtract-consume, never reset: a reset could drop a fast neighbour's credits for
// the next launch):
//   - init: every writer first sends its init (unicast: one to the paired worker of its neighbor; multicast: one to
//     every other device), then waits for the init credits addressed to it and subtracts them, before its first
//     data send. The reader does not wait for init: it fills slots and does the local copy during the round trip.
//   - data_valid: the writer signals cumulative units after the data they cover, on the same sender (FIFO), at the
//     SignalCadence below. The reader waits only while holding no slot.
//   - end: the reader waits for every unit (unicast) or done credit (multicast) it will receive, subtracts them, and
//     barriers its atomics.
//
// Deadlock freedom. The writer's only semaphore wait is init, once, before its first send. Init is sent
// unconditionally at every writer's start, before that writer waits on anything, so it depends on no slot, no
// data_valid and no other wait: every init wait is met once every writer has started. Beyond it, a writer waits only
// for its own reader's slots and for fabric / mux credits, which the fabric drains. The reader waits for data_valid
// only holding no slot, for units its upstream sent in the previous iteration; iteration 0 reads only the input. So by
// induction on the iteration every wait is met.

#include <cstdint>

#include "ag_walk.hpp"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#endif

namespace ag {

// When the writer signals data_valid: after every `every_slots` slots, and always after the last slot of an
// iteration, so a downstream wait target (capped at the iteration's units) is always reached.
class SignalCadence {
public:
    explicit SignalCadence(uint32_t every_slots) : every_(every_slots) {}

    // Accounts one popped slot; returns the units to signal now (0: not yet).
    uint32_t on_slot(const SlotInfo& info) {
        pending_ += info.units;
        if (info.iter_end == 0 && ++slots_ < every_) {
            return 0;
        }
        slots_ = 0;
        const uint32_t units = pending_;
        pending_ = 0;
        return units;
    }

private:
    uint32_t every_, slots_ = 0, pending_ = 0;
};

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

// Walk destination addresses for fabric headers (see the address rule).
struct FabricFrameAddr {
    template <class Acc>
    uint64_t operator()(const Acc& acc, uint32_t page, uint32_t offset) const {
        return tt::tt_fabric::linear::addrgen_detail::get_noc_address(acc, page, offset);
    }
};

// A core's L1 address for a fabric header (semaphores, mirror cores).
FORCE_INLINE uint64_t fabric_frame_addr(uint32_t x, uint32_t y, uint32_t l1) { return safe_get_noc_addr(x, y, l1, 0); }

// A slot's SlotInfo and records are written by the reader's CPU and read by the writer's CPU. CB push_back and
// wait_front are volatile accesses, which do not order ordinary loads and stores (see CoreLocalMem's note), and BH
// RISCs cache L1 reads. So the reader calls slot_publish() between building a slot and push_back, and the writer
// calls slot_acquire() between wait_front and reading the slot.
FORCE_INLINE void slot_publish() { asm volatile("" ::: "memory"); }

FORCE_INLINE void slot_acquire() {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
}

// Waits until the local semaphore reaches `count`, then takes `count` off it. Call Noc::async_atomic_barrier()
// before the kernel ends.
FORCE_INLINE void wait_and_consume(uint32_t sem_l1, uint32_t count) {
    if (count == 0) {
        return;
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1), count);
    noc_semaphore_inc(get_noc_addr(sem_l1), uint32_t{0} - count);
}

// Test-only sync stress (TTNN_ALL_GATHER_FORCE rdelay / wdelay): spins roughly `cycles` cycles; nothing when 0.
template <uint32_t cycles>
FORCE_INLINE void test_delay() {
    if constexpr (cycles != 0) {
        for (volatile uint32_t spin = 0; spin < cycles; spin = spin + 1) {
        }
    }
}

#endif

}  // namespace ag
