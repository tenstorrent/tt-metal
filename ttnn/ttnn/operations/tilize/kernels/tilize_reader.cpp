// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader — NCRISC / NoC0.
//
// One block = one output tile-row x `w` output tile-columns. Blocks are
// linearized `b = wchunk * nt_h + r`; this core owns the contiguous range
// [b0, b0 + nb). Per block the reader issues `tile_h` NoC reads of
// `w * tile_row_bytes` bytes at L1 stride `w * tile_row_bytes` and ONE barrier,
// then pushes `w` tile-sized pages — exactly the contract
// `dataflow_kernel_lib::read_sticks_for_tilize<TILE>` implements, which is why
// the production path is that helper call and nothing else.
//
// `resident == 1` (Refinement 2, A3/C14) is the SECOND production path and the
// only one that is not a NoC read at all: the input shard already sits in this
// core's L1 and `cb_input_sticks` is aliased onto it, so the reader just arms
// the CB. That is what implementing sharding means — re-reading a local shard
// through the TensorAccessor above would re-fetch bytes the core already holds.
//
// The two `if constexpr` arms below are LEVER COUNTERFACTUALS, not alternative
// production paths: `barrier_per_block == 0` is the master.md B7 off-arm (one
// barrier per transaction instead of per block) and `stub_read == 1` is the
// /perf-measure read ablation (keep the CB sync scaffolding, drop the NoC
// payload). At their defaults (1, 0) the compiler emits only the helper call, so
// they cannot perturb the measured path.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t nt_h = get_compile_time_arg_val(1);             // tile-rows
    constexpr uint32_t n_wchunks = get_compile_time_arg_val(2);        // column-blocks per tile-row
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);           // sticks per tile-row
    constexpr uint32_t tile_row_bytes = get_compile_time_arg_val(4);   // 32 * elem
    constexpr uint32_t wt_block = get_compile_time_arg_val(5);         // the block-width knob
    constexpr uint32_t wt_tail = get_compile_time_arg_val(6);
    constexpr uint32_t barrier_per_block = get_compile_time_arg_val(7);  // lever B7 (1 = on)
    constexpr uint32_t stub_read = get_compile_time_arg_val(8);          // ablation (0 = off)
    constexpr uint32_t resident = get_compile_time_arg_val(9);           // A3/C14 zero-copy (1 = on)
    constexpr auto src_args = TensorAccessorArgs<10>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1);
    const uint32_t nb = get_arg_val<uint32_t>(2);

    // A3/C14 zero-copy: `cb_input_sticks` is ALIASED onto this core's own input
    // shard, so the block is already in L1 and there is nothing to fetch — the
    // reader exists only to arm the CB. NO NoC read, no TensorAccessor, and the
    // shard hands us the block width (`wt_block == Wt_shard`), so the whole
    // shard is `nb * wt_block` tile-sized pages.
    if constexpr (resident == 1) {
        const uint32_t pages = nb * wt_block;
        cb_reserve_back(cb_input_sticks, pages);
        cb_push_back(cb_input_sticks, pages);
        return;
    }

    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t i = 0; i < nb; ++i) {
        const uint32_t b = b0 + i;
        const uint32_t wchunk = b / nt_h;      // column-block index
        const uint32_t r = b - wchunk * nt_h;  // global tile-row index

        // The tail column-block is the last one; its width is WT_TAIL (== WT_BLOCK
        // when Wt divides evenly), so the reader's per-block page count matches
        // compute's `WT_BLOCK x n_full` then `WT_TAIL x n_tail` sequence exactly.
        const uint32_t w = (wchunk == n_wchunks - 1) ? wt_tail : wt_block;
        const uint32_t row_bytes = w * tile_row_bytes;
        const uint32_t byte_offset = wchunk * wt_block * tile_row_bytes;

        if constexpr (barrier_per_block == 1 && stub_read == 0) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                src,
                /*total_num_rows*/ tile_h,
                /*row_bytes*/ row_bytes,
                /*start_page*/ r * tile_h,
                /*byte_offset_within_page*/ byte_offset);
        } else {
            // Counterfactual / ablation arm: identical CB accounting (reserve w,
            // read tile_h sticks at L1 stride row_bytes, push w).
            cb_reserve_back(cb_input_sticks, w);
            uint32_t l1_addr = get_write_ptr(cb_input_sticks);
            for (uint32_t s = 0; s < tile_h; ++s) {
                if constexpr (stub_read == 0) {
                    noc_async_read(src.get_noc_addr(r * tile_h + s, byte_offset), l1_addr, row_bytes);
                    if constexpr (barrier_per_block == 0) {
                        noc_async_read_barrier();  // B7 off: one barrier per transaction
                    }
                }
                l1_addr += row_bytes;
            }
            if constexpr (barrier_per_block == 1) {
                noc_async_read_barrier();
            }
            cb_push_back(cb_input_sticks, w);
        }
    }
}
