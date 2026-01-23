// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api.h>
#include <compute_kernel_api/common.h>
#include <compute_kernel_api/eltwise_binary.h>

#include "api/debug/dprint.h"
#include "tt_metal/hw/inc/api/debug/dprint_pages.h"

#include <cstdint>

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_srcA_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_srcB_index = get_compile_time_arg_val(2);
    constexpr uint32_t cb_srcC_index = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out_index = get_compile_time_arg_val(4);

    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_srcA_index, cb_srcB_index, cb_out_index);
    mul_tiles_init(cb_srcA_index, cb_srcB_index);

    for (uint32_t i = 0; i < num_pages; ++i) {
        DPRINT << "Processing page " << i << " / " << num_pages << ENDL();

        DPRINT << "Reading tile A from CB " << cb_srcA_index << ENDL();
        cb_wait_front(cb_srcA_index, 1);
        tt::compute::common::print_full_tile(cb_srcA_index, 0, false);

        DPRINT << "Reading tile B from CB " << cb_srcB_index << ENDL();
        cb_wait_front(cb_srcB_index, 1);
        tt::compute::common::print_full_tile(cb_srcB_index, 0, false);
        tile_regs_acquire();  // math core acquires dst, init dsts to zero
        mul_tiles(cb_srcA_index, cb_srcB_index, 0, 0, dst0);
        tile_regs_commit();

        cb_pop_front(cb_srcA_index, 1);
        cb_pop_front(cb_srcB_index, 1);

        DPRINT << "Reading tile C from CB " << cb_srcC_index << ENDL();
        cb_wait_front(cb_srcC_index, 1);
        tt::compute::common::print_full_tile(cb_srcC_index, 0, false);
        // tile_regs_acquire();
        // // d0 = d0 + c
        // add_tiles_init_with_dt(dst0, cb_srcC_index);
        // add_tiles(cb_srcC_index, dst0, 0, 0, dst0);
        // tile_regs_commit();
        cb_pop_front(cb_srcC_index, 1);

        DPRINT << "Writing output tile to CB " << cb_out_index << ENDL();
        cb_reserve_back(cb_out_index, 1);  // Pack

        tile_regs_wait();                  // Called by packer
        pack_tile(dst0, cb_out_index, 0);  // Pack
        tile_regs_release();               // Pack

        tt::compute::common::print_full_tile(cb_out_index, 0, false);
        cb_push_back(cb_out_index, 1);  // Pack
    }
}  // MAIN
}  // namespace NAMESPACE
