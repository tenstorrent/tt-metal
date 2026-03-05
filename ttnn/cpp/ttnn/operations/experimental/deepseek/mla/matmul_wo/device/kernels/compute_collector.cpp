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
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_out = tt::CBIndex::c_2;
    constexpr auto cb_s2c_out = tt::CBIndex::c_3;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    reconfig_data_format_srca(cb_s2c_in);
    reconfig_data_format_srcb(cb_s2c_in);
    pack_reconfig_data_format(cb_s2c_out);

    binary_op_init_common(cb_s2c_in, cb_s2c_in, cb_s2c_out);

    binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_s2c_in);

    cb_reserve_back(cb_s2c_out, 4);
    uint32_t tile_id = 0;

    for (uint32_t iter_id = 0; iter_id < 4; ++iter_id) {
        cb_wait_front(cb_r2c_w, 1);

        // TODO: Zero out the destination register first.
        tile_regs_acquire();
        zeroacc();
        for (uint32_t k = 0; k < num_cores; ++k) {
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_s2c_in, tile_id++, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s2c_out);
        tile_regs_release();

        cb_pop_front(cb_r2c_w, 1);
    }

    cb_push_back(cb_s2c_out, 4);
}
