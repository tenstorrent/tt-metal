// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"

/*

    1. Router: Matmul: [1, 7168] @ [7168, 256] -> [1, 256]
    2. Sigmoid [1, 256]
    3. Add expert bias [1, 256]
    4. Top-8 operation: [1, 256] -> [1, 8]
    ...
    5. Normalize over [1, 8]

*/

// cb_in0: activations [1, 7168]
// cb_in1: router weights [7168, 256]
// cb_out: router scores [1, 256]
// cb_mm_partials: intermediate buffer for matmul (same as cb_out or separate if untilizing)
template <
    uint32_t cb_in0,
    uint32_t cb_in1,
    uint32_t cb_out,
    uint32_t cb_mm_partials,
    uint32_t in0_block_w,
    uint32_t in0_block_num_tiles,
    uint32_t in1_block_num_tiles,
    uint32_t in1_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    bool untilize_out>
inline void router_compute() {
    constexpr uint32_t mm_out_cb_id = untilize_out ? cb_mm_partials : cb_out;
    constexpr bool in1_transpose_tile = false;

    // Initialize matmul block
    mm_block_init(cb_in0, cb_in1, cb_mm_partials, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

    // Wait for input tiles
    cb_wait_front(cb_in0, in0_block_num_tiles);
    cb_wait_front(cb_in1, in1_block_num_tiles);

    // Acquire tile registers for computation
    tile_regs_acquire();

    // Compute matmul: perform outer product of tiles
    uint32_t dst_index = 0;
    uint32_t in0_index = 0;
    uint32_t in1_index = 0;

    // Iterate over inner dimension (in0_block_w) to accumulate matmul results
    for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
        // Matmul outer product: (out_subblock_h x out_subblock_w) tiles
        // Accumulation is done by iterating matmul_block across inner dimension
        matmul_block(
            cb_in0,
            cb_in1,
            in0_index,
            in1_index,
            dst_index,
            in1_transpose_tile,
            out_subblock_w,
            out_subblock_h,
            in0_block_w);
        in0_index++;               // stride right by 1
        in1_index += in1_block_w;  // stride down by 1
    }

    // Commit tile register writes
    tile_regs_commit();

    // Reserve output buffer and wait for tile registers to be ready
    cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
    tile_regs_wait();

    // Pack tiles from registers to output circular buffer
    uint32_t start_dst_index = 0;
    pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

    // Release tile registers and push output
    tile_regs_release();
    cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

    // Pop consumed input tiles
    cb_pop_front(cb_in0, in0_block_num_tiles);
    cb_pop_front(cb_in1, in1_block_num_tiles);
}
