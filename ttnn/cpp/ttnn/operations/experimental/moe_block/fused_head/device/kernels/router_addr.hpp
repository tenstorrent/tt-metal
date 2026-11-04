// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router-core kernels' addresses, as common runtime args (the program hash then holds none). Every router kernel
// (NCRISC, compute, BRISC) gets the same two common args:
//   [0] the router cores' arena base (the CB range's address on the router cores)
//   [1] the engine-done semaphore address (0 when the logits are on the router cores)
// and the host defines the switches / offsets the macros below build the addresses from:
//   RT_ENG_ON       RT_ENG = common arg 1, only with the logits on the matmul cores
//   RT_MSYNC_OFF    RT_MSYNC = arena base + RT_MSYNC_OFF
// The kernels test RT_ENG with #ifdef: it is defined exactly when its switch is set.
#pragma once

// (get_common_arg_val must be declared before router_rows.hpp, which the kernels include ahead of their API headers)
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/device/kernels/dram_pages.hpp"
#endif

#ifdef RT_ENG_ON
#define RT_ENG (get_common_arg_val<uint32_t>(1))
#endif
#ifdef RT_MSYNC_OFF
#define RT_MSYNC (get_common_arg_val<uint32_t>(0) + RT_MSYNC_OFF)
#endif

// RT_VALID (a padded chunk: fused_head(..., actual_start=, actual_end=), main's chunked-prefill metadata scalars): two
// one-element uint32 DRAM tensors read on device every call (a trace replays with the values of the moment) - the
// chunk's first real token's absolute position and one past its last. The chunk is laid out as main's KV-pad-aware
// rotated block-cyclic layout over the SP rows (moe_padding_config's writer, update_padded_kv_cache): an SP row's real
// rows are a prefix of its rows, their count main's closed form; within the SP row, TP chip p owns rows p S .. (p + 1)
// S. Common args [2] actual_start's address, [3] actual_end's, [4] my SP row r, [5] SP rows, [6] tokens an SP row,
// [7] my first row in it (p S), [8] S. A token t of this chip (0 .. S) is real when t < rt_valid_local(): the planner
// counts, stages and plans only those, and the route words of the rest are zero - a padding token is not routed,
// dispatched, computed or combined.
#if defined(RT_VALID) && !defined(COMPILE_FOR_TRISC)
// the scalars' slots in a scratch from a: its first 64 B aligned address and 64 B on (the DRAM source is 64 B
// aligned; into a 16 B aligned, not 64 B aligned slot the read came back zero) - give it 128 B
FORCE_INLINE uint32_t rt_valid_slot(uint32_t a) { return (a + 63) & ~63u; }
// actual_start's 16 B into l1, actual_end's into l1 + 64 (l1 rt_valid_slot's): issue only - noc_async_read_barrier +
// invalidate_l1_cache before reading them
FORCE_INLINE void rt_valid_issue(uint32_t l1) {
    noc_async_read(dram_pages(get_common_arg_val<uint32_t>(2), 64).get_noc_addr(0), l1, 16);
    noc_async_read(dram_pages(get_common_arg_val<uint32_t>(3), 64).get_noc_addr(0), l1 + 64, 16);
}
// positions in [base, base + len) below end
FORCE_INLINE uint32_t rt_prefix_count(uint32_t end, uint32_t base, uint32_t len) {
    return end <= base ? 0 : (end - base < len ? end - base : len);
}
// this chip's real tokens: its SP row's real rows (moe_padding_config's closed form), less the rows before mine in
// it, clamped to S
FORCE_INLINE uint32_t rt_valid_local(uint32_t l1) {
    const uint32_t start = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    const uint32_t end = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 64);
    const uint32_t my = get_common_arg_val<uint32_t>(4), sp = get_common_arg_val<uint32_t>(5);
    const uint32_t tpc = get_common_arg_val<uint32_t>(6), first = get_common_arg_val<uint32_t>(7);
    const uint32_t s = get_common_arg_val<uint32_t>(8);
    const uint32_t chunk = sp * tpc, slab = start / chunk, bchip = (start / tpc) % sp, boff = start % tpc;
    uint32_t row;
    if (my != bchip) {  // (one run: rows before the boundary row a slab on)
        row = rt_prefix_count(end, ((my < bchip ? slab + 1 : slab) * chunk) + my * tpc, tpc);
    } else {  // (the boundary row starts mid-slab and wraps into the next: two runs)
        row = rt_prefix_count(end, slab * chunk + my * tpc + boff, tpc - boff) +
              rt_prefix_count(end, (slab + 1) * chunk + my * tpc, boff);
    }
    return row <= first ? 0 : (row - first < s ? row - first : s);
}
#endif
