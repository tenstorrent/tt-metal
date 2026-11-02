// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 16-bit blocking L1 load, for draining a store of exactly one halfword.
//
// ckernel::load_blocking accepts a 16-bit operand only from tt-metal
// 2026-07-20 (#50482) on; the pinned tt-metal static_asserts a 32-bit one and
// the kernel fails to build. Same sequence as upstream's 16-bit path: the
// dependent `and` stalls the pipeline until the load returns, spelled per arch
// as upstream does (`x0` destination on Blackhole, the loaded register on
// Wormhole). Wormhole matches a load to a store by start address only, so
// this must name the store's own halfword.
#pragma once

#include <cstdint>

#ifndef tt_l1_ptr
#define tt_l1_ptr
#endif

// The deref stays a C++ access through a tt_l1_ptr pointer off RISC-V so the tt-emule JIT patcher can rebase it;
// the inline asm is the silicon path only.
inline __attribute__((always_inline)) uint16_t load_blocking_u16(volatile tt_l1_ptr uint16_t* ptr) {
#if defined(__riscv)
    uint32_t raw;
    asm volatile(
        "lhu %[raw], (%[ptr])\n\t"
#if defined(ARCH_BLACKHOLE)
        "and x0, x0, %[raw]"
#else
        "and %[raw], %[raw], %[raw]"
#endif
        : [raw] "=r"(raw)
        : [ptr] "r"(ptr)
        : "memory");
    return static_cast<uint16_t>(raw);
#else
    return *ptr;
#endif
}
