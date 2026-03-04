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
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_out = tt::CBIndex::c_2;
    constexpr auto cb_s2c_out = tt::CBIndex::c_3;

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_s2c_in);

    cb_reserve_back(cb_s2c_out, 4);

    for (uint32_t iter_id = 0; iter_id < 4; ++iter_id) {
        cb_wait_front(cb_r2c_w, 1);

        // TODO: Zero out the destination register first.
        tile_regs_acquire();
        for (uint32_t k = 0; k < num_cores; ++k) {
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_s2c_in, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_s2c_out);
        tile_regs_release();

        cb_pop_front(cb_r2c_w, 1);
    }

    cb_push_back(cb_s2c_out, 4);
}
