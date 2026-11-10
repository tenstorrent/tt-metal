// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A kernel's per-core runtime values, read from a DRAM table instead of runtime args (common/args_table.hpp).
//
// Why: runtime args are copied into every chip's dispatch queue on every launch; values that only depend on the plan
// are written once to a table in each chip's DRAM instead, and the kernel reads its own core's block when it starts.
//
// Use: call moe_block_args::load() first thing, then moe_block_args::arg(i) wherever the kernel used
// get_arg_val<uint32_t>(i) (same indices: the block holds the core's args as the host built them). Without
// MOE_BLOCK_ARGS (the kernel's args left as runtime args) both fall back to the runtime args, so a kernel builds either
// way.
//
// The block: the core's args as the host built them, with every address arg holding only its offset, then at word
// MOE_BLOCK_ARGS_FIXUPS_AT the fixups: n, then n entries (arg index | source << 16).
//
// The common args at MOE_BLOCK_ARGS_CA: [0] the table's address, [1] a block's bytes, [2] the DRAM banks, [3] the grid
// width, [4] the L1 address the block lands at (unused with MOE_BLOCK_ARGS_CB), [5] the number of sources S, then the S
// source addresses (all rewritten by the host on a program-cache hit).
//
// What load() does: reads the block of this core (page y * grid width + x of the table, the pages interleaved over the
// DRAM banks) into L1 - CB MOE_BLOCK_ARGS_CB at byte MOE_BLOCK_ARGS_OFF when the op defines it, else the common arg's
// address - then adds each fixup's source address to its arg. Public dataflow APIs only.
//
// arg_pointer(i) / common_pointer(i): where arg i sits in L1 - a kernel reads its fixed args once into a named struct
// and hands helpers a pointer to a variable-length table of args (read in place, never an index into the arg list).
// A kernel whose args hold several blocks (the combine's port-or-accumulator cores) picks its own with set_base().
//
// Also (the MoE head, whose chips differ in common arg values too): moe_block_args::common(i) for
// get_common_arg_val(i) - with MOE_BLOCK_ARGS_CMN the kernel's common args sit in its block from word
// MOE_BLOCK_ARGS_CMN on. A compute kernel cannot read DRAM: the data-movement kernel on its core
// (MOE_BLOCK_ARGS_PEER_*) hands it its args as one page of CB MOE_BLOCK_ARGS_PEER_CB after its own load(), and the
// compute kernel (MOE_BLOCK_ARGS_TRISC) reads them from that page.
#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/cb_api.h"

namespace moe_block_args {
#if defined(MOE_BLOCK_ARGS_TRISC)
// (compute) the page the data-movement kernel pushed: this core's runtime args, its common args from MOE_BLOCK_ARGS_CMN
// on
inline volatile tt_l1_ptr uint32_t* g_block = nullptr;
inline void load() {
    cb_wait_front(MOE_BLOCK_ARGS_PEER_CB, 1);  // (never popped: the page holds the args for the whole launch)
    g_block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(MOE_BLOCK_ARGS_PEER_CB, 0));
}
inline __attribute__((always_inline)) uint32_t arg(uint32_t i) { return g_block[i]; }
inline __attribute__((always_inline)) uint32_t common(uint32_t i) { return g_block[MOE_BLOCK_ARGS_CMN + i]; }
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* arg_pointer(uint32_t i) { return g_block + i; }
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* common_pointer(uint32_t i) {
    return g_block + MOE_BLOCK_ARGS_CMN + i;
}
#else
// (compute, args left as runtime args)
inline void load() {}
inline __attribute__((always_inline)) uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }
inline __attribute__((always_inline)) uint32_t common(uint32_t i) { return get_common_arg_val<uint32_t>(i); }
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* arg_pointer(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(i));
}
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* common_pointer(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_addr(i));
}
#endif
}  // namespace moe_block_args

#else
#include "api/dataflow/dataflow_api.h"

namespace moe_block_args {

#if defined(MOE_BLOCK_ARGS)
// (see the top of this file)
inline volatile tt_l1_ptr uint32_t* g_block = nullptr;
inline uint32_t g_base = 0;

inline void load() {
    constexpr uint32_t a = MOE_BLOCK_ARGS_CA;
    const uint32_t table = get_common_arg_val<uint32_t>(a), block = get_common_arg_val<uint32_t>(a + 1);
    const uint32_t banks = get_common_arg_val<uint32_t>(a + 2), gx = get_common_arg_val<uint32_t>(a + 3);
#ifdef MOE_BLOCK_ARGS_CB
    const uint32_t l1 = get_write_ptr(MOE_BLOCK_ARGS_CB) + MOE_BLOCK_ARGS_OFF;
#else
    const uint32_t l1 = get_common_arg_val<uint32_t>(a + 4);
#endif
    const uint32_t page = get_absolute_logical_y() * gx + get_absolute_logical_x();
    noc_async_read(get_noc_addr_from_bank_id<true>(page % banks, table + (page / banks) * block), l1, block);
    noc_async_read_barrier();
    g_block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    g_base = 0;
    const uint32_t n = g_block[MOE_BLOCK_ARGS_FIXUPS_AT];
    for (uint32_t k = 0; k < n; ++k) {  // (this launch's buffer addresses onto their args' offsets)
        const uint32_t e = g_block[MOE_BLOCK_ARGS_FIXUPS_AT + 1 + k];
        g_block[e & 0xFFFF] += get_common_arg_val<uint32_t>(a + 6 + (e >> 16));
    }
#ifdef MOE_BLOCK_ARGS_PEER_CB
    // (the compute kernel's args, from word MOE_BLOCK_ARGS_PEER_AT of this block: one page of its CB, fixups applied)
    cb_reserve_back(MOE_BLOCK_ARGS_PEER_CB, 1);
    volatile tt_l1_ptr uint32_t* peer =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(MOE_BLOCK_ARGS_PEER_CB));
    for (uint32_t k = 0; k < MOE_BLOCK_ARGS_PEER_N; ++k) {
        peer[k] = g_block[MOE_BLOCK_ARGS_PEER_AT + k];
    }
    cb_push_back(MOE_BLOCK_ARGS_PEER_CB, 1);
#endif
}

FORCE_INLINE void set_base(uint32_t off) { g_base = off; }
FORCE_INLINE uint32_t arg(uint32_t i) { return g_block[g_base + i]; }
FORCE_INLINE volatile tt_l1_ptr uint32_t* arg_pointer(uint32_t i) { return g_block + g_base + i; }
#ifdef MOE_BLOCK_ARGS_CMN
FORCE_INLINE uint32_t common(uint32_t i) { return g_block[MOE_BLOCK_ARGS_CMN + i]; }
FORCE_INLINE volatile tt_l1_ptr uint32_t* common_pointer(uint32_t i) { return g_block + MOE_BLOCK_ARGS_CMN + i; }
#else
FORCE_INLINE uint32_t common(uint32_t i) { return get_common_arg_val<uint32_t>(i); }
FORCE_INLINE volatile tt_l1_ptr uint32_t* common_pointer(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_addr(i));
}
#endif

#else
// (args left as runtime args: the same indices; set_base offsets them for a kernel with several blocks)
inline uint32_t g_base = 0;
inline void load() {}
FORCE_INLINE void set_base(uint32_t off) { g_base = off; }
FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(g_base + i); }
FORCE_INLINE uint32_t common(uint32_t i) { return get_common_arg_val<uint32_t>(i); }
FORCE_INLINE volatile tt_l1_ptr uint32_t* arg_pointer(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(g_base + i));
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* common_pointer(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_addr(i));
}
#endif

}  // namespace moe_block_args
#endif
