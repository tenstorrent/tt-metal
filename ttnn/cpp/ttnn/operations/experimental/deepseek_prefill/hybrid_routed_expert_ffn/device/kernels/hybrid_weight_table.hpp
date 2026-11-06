// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Per-expert weight base addresses from one DRAM table instead of per-core runtime args.
//
// Each expert costs the two halves' data-movement kernels nine runtime-arg words per core, and
// per-core runtime args sit in the kernel-config ring this op is short of -- 1.1 KB of it at 32
// experts. The table is [gate[0..E), up[0..E), down[0..E)], one page; the union reader copies it
// into an L1 page past both halves' buffers before the first pass and the writer waits for that
// page, so every half reads its addresses from L1 for the whole kernel.
//
// Enabled by the program factory defining HYB_WEIGHT_TABLE. Without it each half reads its own
// runtime args, exactly as it does alone.

#pragma once

#include <cstdint>

#ifdef HYB_WEIGHT_TABLE

#if !defined(HYB_WEIGHT_TABLE_CB) || !defined(HYB_WEIGHT_TABLE_BYTES)
#error "HYB_WEIGHT_TABLE needs HYB_WEIGHT_TABLE_CB and HYB_WEIGHT_TABLE_BYTES from the program factory"
#endif

enum HybWeightRole : uint32_t { HYB_W_GATE = 0, HYB_W_UP = 1, HYB_W_DOWN = 2 };

inline uint32_t hyb_weight_addr(uint32_t role, uint32_t experts, uint32_t e) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(HYB_WEIGHT_TABLE_CB))[role * experts + e];
}

#if !defined(COMPILE_FOR_TRISC)
#ifdef HYB_WEIGHT_TABLE_RT
// The reader's half of the handoff: land the table, then publish the page. The table is one page
// of an interleaved buffer, so it sits in bank 0 at the buffer's base.
inline void hyb_load_weight_table() {
    cb_reserve_back(HYB_WEIGHT_TABLE_CB, 1);
    noc_async_read(
        get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(HYB_WEIGHT_TABLE_RT)),
        get_write_ptr(HYB_WEIGHT_TABLE_CB),
        HYB_WEIGHT_TABLE_BYTES);
    noc_async_read_barrier();
    cb_push_back(HYB_WEIGHT_TABLE_CB, 1);
}
#endif

// The writer's half: the page is published once and never popped.
inline void hyb_wait_weight_table() { cb_wait_front(HYB_WEIGHT_TABLE_CB, 1); }
#endif

#endif  // HYB_WEIGHT_TABLE
