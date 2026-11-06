// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router-core kernels' addresses, as common runtime args (the program hash then holds none). Every router kernel
// (NCRISC, compute, BRISC) gets the same two common args:
//   [0] the router cores' arena base (the CB range's address on the router cores)
//   [1] the engine-done semaphore address (the matmul cores count their partial logits tiles into it): RT_ENG
// and the host defines the offset the macro below builds an address from:
//   RT_MSYNC_OFF    RT_MSYNC = arena base + RT_MSYNC_OFF
#pragma once

// (get_common_arg_val must be declared before router_rows.hpp, which the kernels include ahead of their API headers)
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/ra_accessors.hpp"
#endif

#define RT_ENG (get_common_arg_val<uint32_t>(1))
#ifdef RT_MSYNC_OFF
#define RT_MSYNC (get_common_arg_val<uint32_t>(0) + RT_MSYNC_OFF)
#endif

// RT_VALID (a padded forward), read on device every call (a trace replays with the values of the moment):
//  * fused_head(..., actual_start=, actual_end=): one chunk, main's chunked-prefill metadata scalars - two one-element
//    uint32 DRAM tensors, the chunk's first real token's absolute position and one past its last (common [2], [3]);
//  * RT_META, fused_head(..., metadata=): the forward's metadata words (common/prefill/prefill_chunk.py, one uint32
//    DRAM tensor, common [2]) - RT_FWD (forward_chunks) chunks packed in the forward: at 1 (slot, start, end), above
//    {num_chunks, 0} then RT_FWD (slot, start, end) records (past num_chunks unused, empty).
// Each SP row holds every chunk's B = [6] / RT_FWD rows back to back (chunk k: rows k B ..), each chunk laid out as
// main's KV-pad-aware rotated block-cyclic layout over the SP rows (moe_padding_config's writer,
// update_padded_kv_cache): an SP row's real rows of chunk k are a prefix of its block, their count main's closed form.
// Within the SP row, TP chip p owns rows p S .. (p + 1) S. Common args [4] my SP row r, [5] SP rows, [6] tokens an SP
// row, [7] my first row in it (p S; with token loops the loop's first row plus p S), [8] S, [9] a token loop's rows,
// [10] the loops, [11] the first loop's first row of mine. A token of this chip is real when its row is
// (rt_row_real): the planner counts, stages and plans only those, and the route words of the rest are zero - a
// padding token is not routed, dispatched, computed or combined.
#if defined(RT_VALID) && !defined(COMPILE_FOR_TRISC)
#ifndef RT_FWD
#define RT_FWD 1
#endif
static_assert(RT_FWD >= 1 && RT_FWD <= 8, "a forward of 1 .. 8 chunks (the metadata's 104 B in a 128 B slot)");
// the metadata's slot in a scratch from a: its first 64 B aligned address and 64 B on (the DRAM source is 64 B
// aligned; into a 16 B aligned, not 64 B aligned slot the read came back zero) - give it 192 B
FORCE_INLINE uint32_t rt_valid_slot(uint32_t a) { return (a + 63) & ~63u; }
// issue only - noc_async_read_barrier + invalidate_l1_cache before reading them: the metadata words into l1, or
// actual_start's 16 B into l1 and actual_end's into l1 + 64
FORCE_INLINE void rt_valid_issue(uint32_t l1) {
#ifdef RT_META
    constexpr uint32_t bytes = RT_FWD == 1 ? 16 : (4 * (2 + 3 * RT_FWD) + 15) / 16 * 16;
    noc_async_read(TensorAccessor(ra_acc::start, get_common_arg_val<uint32_t>(2), 64).get_noc_addr(0), l1, bytes);
#else
    noc_async_read(TensorAccessor(ra_acc::start, get_common_arg_val<uint32_t>(2), 64).get_noc_addr(0), l1, 16);
    noc_async_read(TensorAccessor(ra_acc::end, get_common_arg_val<uint32_t>(3), 64).get_noc_addr(0), l1 + 64, 16);
#endif
}
// positions in [base, base + len) below end
FORCE_INLINE uint32_t rt_prefix_count(uint32_t end, uint32_t base, uint32_t len) {
    return end <= base ? 0 : (end - base < len ? end - base : len);
}
// SP row my's real rows of a chunk [start, end) of tpc rows an SP row (moe_padding_config's closed form)
FORCE_INLINE uint32_t rt_chunk_rows(uint32_t start, uint32_t end, uint32_t my, uint32_t sp, uint32_t tpc) {
    const uint32_t chunk = sp * tpc, slab = start / chunk, bchip = (start / tpc) % sp, boff = start % tpc;
    if (my != bchip) {  // (one run: rows before the boundary row a slab on)
        return rt_prefix_count(end, ((my < bchip ? slab + 1 : slab) * chunk) + my * tpc, tpc);
    }
    // (the boundary row starts mid-slab and wraps into the next: two runs)
    return rt_prefix_count(end, slab * chunk + my * tpc + boff, tpc - boff) +
           rt_prefix_count(end, (slab + 1) * chunk + my * tpc, boff);
}
// my SP row's real rows of each chunk of the forward (its block's real prefix), from the words at l1
struct RtRows {
    uint32_t B;          // an SP row's rows a chunk
    uint32_t v[RT_FWD];  // chunk k's real rows: rows k B .. k B + v[k] of the SP row
    // is row `row` of my SP row real (rows past every chunk's - the block's padding rows - are not)
    FORCE_INLINE bool real(uint32_t row) const {
        const uint32_t k = row / B;
        return k < RT_FWD && row - k * B < v[k];
    }
    // the real rows in [f, f + n)
    FORCE_INLINE uint32_t count(uint32_t f, uint32_t n) const {
        uint32_t c = 0;
        for (uint32_t k = 0; k < RT_FWD; ++k) {
            const uint32_t a = k * B, b = a + v[k];
            const uint32_t lo = a > f ? a : f, hi = b < f + n ? b : f + n;
            c += hi > lo ? hi - lo : 0;
        }
        return c;
    }
};
FORCE_INLINE RtRows rt_rows(uint32_t l1) {
    RtRows r;
    const uint32_t my = get_common_arg_val<uint32_t>(4), sp = get_common_arg_val<uint32_t>(5);
    r.B = get_common_arg_val<uint32_t>(6) / RT_FWD;
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
#ifdef RT_META
    const uint32_t n = RT_FWD == 1 ? 1u : w[0];  // (the chunks in use: the rest are empty)
    for (uint32_t k = 0; k < RT_FWD; ++k) {
        const uint32_t o = RT_FWD == 1 ? 1u : 3u + 3u * k;
        r.v[k] = k < n ? rt_chunk_rows(w[o], w[o + 1], my, sp, r.B) : 0;
    }
#else
    r.v[0] = rt_chunk_rows(w[0], w[16], my, sp, r.B);
#endif
    return r;
}
#endif
