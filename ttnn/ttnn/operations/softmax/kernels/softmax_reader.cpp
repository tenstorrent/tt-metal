// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax reader.
//
// Per-core work:
//   1. One-shot at boot: emit one MAX scaler tile (value 1.0) into cb_max_scaler
//      and one SUM scaler tile (value 1.0) into cb_sum_scaler. Both scalers
//      use the pool-type/reduce-dim-aware overload of
//      `calculate_and_prepare_reduce_scaler` so the tile fill pattern is
//      correct for both REDUCE_ROW (col-0 fill for SUM/AVG, row-0 fill for MAX)
//      and REDUCE_COL (row-0 fill).
//   2. Strip loop: for each of `num_strips` strips assigned to this core,
//      compute the per-strip tile-id pattern and push `reduce_dim_tiles`
//      fp32 tiles into cb_input_tiles in the order the compute kernel expects.
//
// Strip-to-tile mapping (input tiles are interleaved DRAM, row-major
// indexed; tile index = nc*Ht*Wt + ht*Wt + wt):
//   dim=-1 (REDUCE_ROW): strip = 1 × Wt tiles
//       strip s spans nc = s / Ht, ht = s % Ht, wt = 0..Wt-1.
//       Equivalently, starting tile id = s * Wt, stride = 1.
//   dim=-2 (REDUCE_COL): strip = Ht × 1 tiles
//       strip s spans nc = s / Wt, wt = s % Wt, ht = 0..Ht-1.
//       Starting tile id = (s / Wt) * Ht * Wt + (s % Wt), stride = Wt.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_max_scaler = 8;
constexpr uint32_t cb_sum_scaler = 9;
}  // namespace

void kernel_main() {
    constexpr uint32_t dim_is_row = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_dim_tiles = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_strips = get_arg_val<uint32_t>(1);
    const uint32_t start_strip = get_arg_val<uint32_t>(2);

    // -------- One-shot scaler fills --------
    // MAX scaler value: 1.0 (the LLK uses the scaler tile as SrcB; for MAX it
    // gates which positions contribute — 1.0 means "include this element").
    // SUM scaler value: 1.0 (no pre-division — the SUM is later inverted by
    // the recip postop in Phase C).
    //
    // The pool-type/reduce-dim-aware overload picks the correct tile fill
    // pattern (col-0 vs row-0) per (pool, reduce_dim) combo. Using the legacy
    // single-template-arg overload would silently produce wrong values for
    // REDUCE_COL or SUM REDUCE_ROW (matmul-path).
    if constexpr (dim_is_row != 0) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_max_scaler,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_ROW>();
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_sum_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>();
    } else {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_max_scaler,
            ckernel::PoolType::MAX,
            ckernel::ReduceDim::REDUCE_COL>();
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_sum_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_COL>();
    }

    // -------- Strip streaming loop --------
    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    for (uint32_t i = 0; i < num_strips; ++i) {
        const uint32_t s = start_strip + i;

        // Compute the strip's first tile id and per-tile stride.
        uint32_t base_tile_id;
        uint32_t stride;
        if constexpr (dim_is_row != 0) {
            // dim = -1, REDUCE_ROW: strip = 1 × Wt, contiguous tiles.
            base_tile_id = s * Wt;
            stride = 1;
        } else {
            // dim = -2, REDUCE_COL: strip = Ht × 1, stride Wt in tile space.
            const uint32_t nc = s / Wt;
            const uint32_t wt = s - nc * Wt;
            base_tile_id = nc * (Ht * Wt) + wt;
            stride = Wt;
        }

        // Read the whole strip into cb_input_tiles. Phase A in compute uses
        // WaitUpfrontNoPop on this CB, so the entire strip must be resident
        // before the reduce starts. The CB is sized to 2 × reduce_dim_tiles
        // so the next strip's reader fill can pipeline against the current
        // strip's compute.
        cb_reserve_back(cb_input_tiles, reduce_dim_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
        for (uint32_t t = 0; t < reduce_dim_tiles; ++t) {
            const uint32_t tile_id = base_tile_id + t * stride;
            noc_async_read_tile(tile_id, accessor, l1_write_addr);
            l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, reduce_dim_tiles);
    }
}
