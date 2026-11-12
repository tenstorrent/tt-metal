// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#if !defined(ARCH_QUASAR)
// ckernel::load_blocking (the store drain below) is WH/BH only: Quasar's ckernel.h has no load_blocking,
// and from a data-movement build it #errors unless COMPILE_FOR_TRISC is defined. The Quasar branch uses a
// fence instead. Same guard as data_movement/common/kernels/common.hpp.
#include "ckernel.h"
#endif

union value {
    float f;
    uint32_t u;
};
constexpr uint32_t onepage = 1;

// Zeroes the first `bytes` of the caller's dataflow buffer at its current write cursor.
// Takes the buffer the caller already holds rather than a handle: one buffer must be driven by a
// single DataflowBuffer object, so the caller's object is passed through instead of constructing a
// second one for the same FIFO.
template <typename DFB>
inline void zero_buffer(const DFB& dfb, uint32_t bytes) {
    Noc noc;
    noc.async_write_zeros(dfb, bytes);
    noc.write_zeros_l1_barrier();
}

// The non-zero fill is baby-RISCV stores and the page is then handed to the NoC as the source of
// every output write. A store can retire before its write-request reaches L1, and the RISCV core
// and the NoC are different L1 clients with no program-order guarantee between them
// (WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md). Read back the last element of the page: the
// RISCV's L1 port processes its requests in order, so once that load returns every fill store has
// landed. Same construction as #50374 (deepseek_grouped_gate) and the pad reader's loop-back guard.
// The load goes through the fill's own element type so it hits the last store's start address for
// every dtype, naturally aligned and never below the page base: Wormhole holds a load behind a queued
// store only when the start addresses are equal, Blackhole on any byte overlap.
// On Quasar DM the dataflow buffer write pointer is the uncached alias, so the fill already targets
// TL1; the fence orders those stores before the NoC command that reads the page.
template <class T>
inline void drain_fill_stores(T* ptr, uint32_t elems) {
#if defined(ARCH_QUASAR)
    (void)ptr;
    (void)elems;
    __asm__ __volatile__("fence" ::: "memory");
#else
    (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr T*>(ptr + elems - 1));
#endif
}
