// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// riscv_transpose bake-off: cross-RISC counters for the multi-RISC transpose. Each counter is the
// tiles_received stream register of an otherwise unused CB index: one writer RISC per counter, any
// number of readers, and the firmware zeroes every CB stream register between launches, so a launch
// always starts from 0 (no L1 flag to initialize race-free). Works the same on BRISC, NCRISC and TRISCs.
// Raw sync (no helper): kernel_lib has no primitive for "N producers on different RISCs, one consumer"
// over plain L1 data; the CB API is single-producer / single-consumer and its compute side is split
// across the UNPACK and PACK threads.
namespace rvt {

FORCE_INLINE uint32_t ctr_read(uint32_t cb) { return *get_cb_tiles_received_ptr(cb) & 0xFFFFu; }

FORCE_INLINE void ctr_wait(uint32_t cb, uint32_t target) {
    while (ctr_read(cb) < target) {
    }
}

// Wrap-safe (16-bit counter): returns once the counter is at or past target (mod 2^16, within 2^15).
FORCE_INLINE void ctr_wait16(uint32_t cb, uint32_t target) {
    while (static_cast<int16_t>(static_cast<uint16_t>(ctr_read(cb) - target)) < 0) {
    }
}

// Every earlier L1 store of this RISC is complete before the counter moves.
FORCE_INLINE void ctr_post(uint32_t cb, uint32_t value) {
    asm volatile("fence" ::: "memory");
    *get_cb_tiles_received_ptr(cb) = value;
}

// Drops this RISC's data-cache lines (Blackhole: 4 x 16 B, write-through) before reading L1 that
// another agent (NoC, another RISC) wrote.
FORCE_INLINE void l1_invalidate() { asm volatile("fence" ::: "memory"); }

}  // namespace rvt
