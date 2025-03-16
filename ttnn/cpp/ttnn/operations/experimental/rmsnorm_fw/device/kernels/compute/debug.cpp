// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"

#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t block = get_compile_time_arg_val(1);
    constexpr uint32_t num_inner = get_compile_time_arg_val(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_rms_intermediate = tt::CBIndex::c_5;
    constexpr auto cb_rms_output = tt::CBIndex::c_7;

    constexpr uint32_t onetile = 1;

    init_sfpu(cb_input, cb_rms_output);
    binary_op_init_common(cb_input, cb_input, cb_rms_output);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_input, 1);

        tile_regs_acquire();
        cb_reserve_back(cb_rms_intermediate, onetile);
        copy_tile_init(cb_input);
        copy_tile(cb_input, /* tile_idx */ 0, /* register_idx */ 0);

        mul_binary_tile_init();
        mul_binary_tile(0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rms_intermediate);
        cb_push_back(cb_rms_intermediate, onetile);
        tile_regs_release();

        {
            cb_wait_front(cb_rms_intermediate, onetile);
            tile_regs_acquire();
            cb_reserve_back(cb_rms_output, onetile);
            copy_tile_init(cb_rms_intermediate);
            copy_tile(cb_rms_intermediate, 0, 0);

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_rms_output);
            cb_push_back(cb_rms_output, onetile);
            tile_regs_release();
        }

        cb_pop_front(cb_input, 1);
        cb_pop_front(cb_rms_intermediate, 1);
    }

    DPRINT << "All done" << ENDL();
}

}  // namespace NAMESPACE
