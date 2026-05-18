// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for backward_softmax (VJP of softmax).
//
// Drains cb_grad_input one tile at a time and writes each tile to DRAM at the
// same logical tile_id as the corresponding grad_output tile that produced it
// (output shape == input shape; same tile_id arithmetic as the reader's pass-2
// stream).
//
// Per lane: NUM_BLOCKS * BLOCK_SIZE tiles. Compute pushes them in
// pass-2 order: block 0 tile 0..BLOCK_SIZE-1, block 1, ..., block NUM_BLOCKS-1.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // RT args (per-core under multi-core distribution): [start_lane, num_lanes]
    // is this core's contiguous lane slice. See the reader kernel for context.
    uint32_t grad_input_addr = get_arg_val<uint32_t>(0);
    uint32_t start_lane = get_arg_val<uint32_t>(1);
    uint32_t num_lanes = get_arg_val<uint32_t>(2);

    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr uint32_t DIM_IS_W = get_compile_time_arg_val(2);  // 1 = dim=-1, 0 = dim=-2
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

    constexpr uint32_t cb_grad_input = 16;

    const uint32_t tile_bytes = get_tile_size(cb_grad_input);
    const auto accessor = TensorAccessor(dst_args, grad_input_addr, tile_bytes);

    constexpr uint32_t reduce_lanes_per_nc = DIM_IS_W ? Ht : Wt;

    for (uint32_t lane_idx = 0; lane_idx < num_lanes; ++lane_idx) {
        const uint32_t lane = start_lane + lane_idx;
        const uint32_t nc = lane / reduce_lanes_per_nc;
        const uint32_t idx = lane % reduce_lanes_per_nc;
        const uint32_t tile_id_origin = nc * Ht * Wt;

        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            for (uint32_t k = 0; k < BLOCK_SIZE; ++k) {
                uint32_t tile_id;
                if constexpr (DIM_IS_W) {
                    tile_id = tile_id_origin + idx * Wt + (b * BLOCK_SIZE + k);
                } else {
                    tile_id = tile_id_origin + (b * BLOCK_SIZE + k) * Wt + idx;
                }

                cb_wait_front(cb_grad_input, 1);
                const uint32_t l1_read_addr = get_read_ptr(cb_grad_input);
                noc_async_write_tile(tile_id, accessor, l1_read_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_grad_input, 1);
            }
        }
    }
}
