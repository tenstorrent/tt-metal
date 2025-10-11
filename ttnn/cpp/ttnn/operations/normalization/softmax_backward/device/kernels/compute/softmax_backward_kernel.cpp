// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include <compute_kernel_api/common.h>
#include <compute_kernel_api/tile_move_copy.h>
#include <compute_kernel_api/eltwise_binary.h>
#include <compute_kernel_api/reduce.h>

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);       // softmax_output (y)
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);       // upstream_grad (grad)
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);        // output
    constexpr uint32_t intermed0_cb_id = get_compile_time_arg_val(3);  // y * grad
    constexpr uint32_t intermed1_cb_id = get_compile_time_arg_val(4);  // sum(y * grad)
    constexpr uint32_t intermed2_cb_id = get_compile_time_arg_val(5);  // grad - sum(y * grad)
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(6);

    // Runtime args
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Initialize compute
    binary_op_init_common(src0_cb_id, src1_cb_id, out_cb_id);
    mul_tiles_init(src0_cb_id, src1_cb_id);

    // Process each tile individually
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Step 1: Compute y * grad (element-wise multiplication)
        cb_reserve_back(out_cb_id, 1);
        cb_wait_front(src0_cb_id, 1);  // softmax_output
        cb_wait_front(src1_cb_id, 1);  // upstream_grad

        tile_regs_acquire();
        // Multiply y * grad
        mul_tiles(src0_cb_id, src1_cb_id, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, out_cb_id);
        tile_regs_release();

        cb_pop_front(src0_cb_id, 1);
        cb_pop_front(src1_cb_id, 1);
        cb_push_back(out_cb_id, 1);
    }
}
}  // namespace NAMESPACE
