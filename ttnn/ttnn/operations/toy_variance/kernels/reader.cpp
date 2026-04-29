// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for toy_variance.
//
// Variance via Var(x) = E[(x - E[x])^2] is a two-pass algorithm:
//   Pass 1: stream x  → compute mean
//   Pass 2: stream x  → compute (x - mean)^2 → mean
//
// Per-row two-pass shape: for each ht, stream that one row twice (pass 1
// then pass 2) before moving on. Compute consumes one row at a time, so
// cb_mean / cb_variance / cb_in only ever hold a single row's working set —
// L1 footprint is independent of Ht.
//
// Tile-id ordering and per-tile push are owned by stream_axis_blocks. The
// caller's outer loop chooses the slice (and pass count); the helper walks
// reduce-axis blocks in order:
//   for outer in [preserved_start, preserved_end): for b: for wt:
//     tile_id = outer * Wt + b * BLOCK_SIZE + wt
//
// preserved_start / preserved_end come from runtime args. Single-core today
// uses (0, Ht); a multicore extension is purely a descriptor change — each
// core gets its own row range, no kernel changes needed.
//
// The scaler tile (1/N for SUM-reduce-as-mean) is pushed once at startup.
// reduce<> waits on it but never pops, so the same tile serves both passes
// for every row.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t preserved_start = get_arg_val<uint32_t>(1);
    uint32_t preserved_end = get_arg_val<uint32_t>(2);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);  // 1/N as fp32 bits
    constexpr uint32_t has_partial_w = get_compile_time_arg_val(4);
    constexpr uint32_t partial_w = get_compile_time_arg_val(5);  // valid positions in last W-tile
    constexpr auto src_args = TensorAccessorArgs<6>();

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 2;

    // Scaler = 1/N → SUM reduce produces means directly. For non-tile-aligned
    // W, also emit a partial scaler tile that zeros out positions beyond
    // partial_w; the compute kernel selects it for the last W-tile of the
    // last block via ReducePartialScaler::last_tile_at(1).
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    if constexpr (has_partial_w) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            partial_w>(scaler_f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f);
    }

    uint32_t tile_bytes = get_tile_size(cb_in);
    const auto accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    // Per-row two-pass: stream one row twice, then move to the next. Compute
    // processes each row end-to-end before its working set is reused.
    for (uint32_t outer = preserved_start; outer < preserved_end; ++outer) {
        // Pass 1: feeds accumulate_reduce → cb_mean (one tile produced per row).
        dataflow_kernel_lib::stream_axis_blocks<cb_in, ckernel::ReduceDim::REDUCE_ROW>(
            accessor, outer, outer + 1, Ht, Wt, BLOCK_SIZE);
        // Pass 2: feeds sub<COL> + square + accumulate_reduce_block → cb_variance.
        dataflow_kernel_lib::stream_axis_blocks<cb_in, ckernel::ReduceDim::REDUCE_ROW>(
            accessor, outer, outer + 1, Ht, Wt, BLOCK_SIZE);
    }
}
