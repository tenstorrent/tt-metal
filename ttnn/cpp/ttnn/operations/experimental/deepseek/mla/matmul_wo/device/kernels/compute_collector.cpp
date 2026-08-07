// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto collector_core_id = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_s2c_in2_id = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out_id = tt::CBIndex::c_4;

    CircularBuffer cb_s2c_in2(cb_s2c_in2_id);
    CircularBuffer cb_s2c_out(cb_s2c_out_id);

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    constexpr uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    reconfig_data_format_srca(cb_s2c_in2_id);
    reconfig_data_format_srcb(cb_s2c_in2_id);
    pack_reconfig_data_format(cb_s2c_out_id);

    binary_op_init_common(cb_s2c_in2_id, cb_s2c_in2_id, cb_s2c_out_id);

    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_s2c_in2_id);

    cb_s2c_out.reserve_back(num_iters);

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        cb_s2c_in2.wait_front(num_cores);

        tile_regs_acquire();
        for (uint32_t k = 0; k < num_cores; ++k) {
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_s2c_in2_id, k, 0 /*dst_tile_index*/);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_s2c_out_id);
        tile_regs_release();

        cb_s2c_in2.pop_front(num_cores);
    }

    cb_s2c_out.push_back(num_iters);
}
