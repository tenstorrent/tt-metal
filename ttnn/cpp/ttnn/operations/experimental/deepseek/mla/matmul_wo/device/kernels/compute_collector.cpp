// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto collector_core_id = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    constexpr uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    reconfig_data_format_srca(cb_s2c_in2);
    reconfig_data_format_srcb(cb_s2c_in2);
    pack_reconfig_data_format(cb_s2c_out);

    binary_op_init_common(cb_s2c_in2, cb_s2c_in2, cb_s2c_out);

    binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_s2c_in2);

    cb_reserve_back(cb_s2c_out, num_iters);

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        cb_wait_front(cb_s2c_in2, num_cores);

        tile_regs_acquire();
        for (uint32_t k = 0; k < num_cores; ++k) {
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_s2c_in2, k, 0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_s2c_out);
        tile_regs_release();

        cb_pop_front(cb_s2c_in2, num_cores);
    }

    cb_push_back(cb_s2c_out, num_iters);
}
