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
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/rt_rows.hpp"
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
FORCE_INLINE RtRows rt_rows(uint32_t l1) {
    return rt_rows_of(
        l1, get_common_arg_val<uint32_t>(4), get_common_arg_val<uint32_t>(5), get_common_arg_val<uint32_t>(6));
}
#endif
