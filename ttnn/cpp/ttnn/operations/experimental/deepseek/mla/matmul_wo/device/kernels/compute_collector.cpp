// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "llk_defs.h"
#include "matmul_wo_ring_common.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reduce.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;

    // CB Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_1;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_s2c_in, cb_s2c_in, cb_c2s_out);

    for (uint32_t iter_id = 0; iter_id < 4; ++iter_id) {
        cb_wait_front(cb_c2w_rdy, 1);

        tile_regs_acquire();
        // Reduce 12 partial sums into 1 tile
        // for (uint32_t k = 0; k < 12; ++k) {
        //     reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_c2w_rdy, cb_c2w_rdy, 0, 0, 0);
        // }

        cb_pop_front(cb_c2w_rdy, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c2s_out);
        tile_regs_release();
    }
    reduce_uninit();
}
