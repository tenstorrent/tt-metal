// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// atan_mean — Reader kernel.
//
// Once at program startup: writes a single ``1/W`` scaler tile to ``cb_scaler``
// (bf16, matmul col-0 fill — AVG/REDUCE_ROW uses the matmul reduce path) via
// ``calculate_and_prepare_reduce_scaler``. The scaler is never popped by the
// compute kernel — every ``reduce<>`` call re-waits on it.
//
// Per row-tile: streams ``Wt`` float32 input tiles from DRAM into
// ``cb_input_tiles`` in tile-id order. The tile-id formula for tiled
// ``(N, C, H, W)`` interleaved tensors is:
//     input_tile_id = r * Wt + wt
// where ``r`` is the global row-tile index (over ``(N, C, Ht)``).
//
// CT args: [CB_INPUT_TILES, CB_SCALER, W, Wt, TensorAccessorArgs...]
// RT args: [src_addr, num_row_tiles, start_row_tile]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_row_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_row_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();

    // Emit the 1/W scaler tile in matmul col-0 layout. AVG + REDUCE_ROW uses
    // the matmul-based reduce path internally (see reduce_helpers_common.hpp);
    // the pool-type-aware overload picks the correct fill pattern.
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, W>();

    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto src_accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    const uint32_t end_row_tile = start_row_tile + num_row_tiles;
    for (uint32_t r = start_row_tile; r < end_row_tile; ++r) {
        const uint32_t base_tile = r * Wt;
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_reserve_back(cb_input_tiles, 1);
            const uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
            noc_async_read_tile(base_tile + wt, src_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, 1);
        }
    }
}
