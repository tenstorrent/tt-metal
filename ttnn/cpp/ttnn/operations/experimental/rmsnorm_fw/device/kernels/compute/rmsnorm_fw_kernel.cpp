// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
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
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_eps = tt::CBIndex::c_3;
    constexpr auto cb_gamma = tt::CBIndex::c_4;
    constexpr auto cb_rms_intermediate = tt::CBIndex::c_5;
    constexpr auto cb_output = tt::CBIndex::c_6;
    constexpr auto cb_rms_output = tt::CBIndex::c_7;

    constexpr uint32_t onetile = 1;

    DPRINT << "Num inner: " << num_inner << ENDL();

    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);

#ifdef DO_MASK_W
    cb_wait_front(cb_mask, onetile);
#endif

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_input, num_inner);

        const uint32_t accum_dst = 2U;
        const uint32_t dst0 = 0;
        // cb_reserve_back(cb_rms_intermediate, onetile);

        tile_regs_acquire();
        // cb_wait_front(cb_input, onetile);  // comes from the reader
        cb_reserve_back(cb_rms_intermediate, onetile);

        copy_tile_init(cb_input);
        copy_tile(cb_input, 0, dst0);
        cb_pop_front(cb_input, onetile);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(dst0, cb_rms_intermediate);
        cb_push_back(cb_rms_intermediate, onetile);
        tile_regs_release();

        // tile_regs_acquire();
        // for (uint32_t col = 0; col < num_inner; ++col) {
        //     copy_tile_init(cb_input);
        //     copy_tile(cb_input, /* tile_idx */ col, /* register_idx */ 0);

        // #ifdef DO_MASK_W
        // if (col + 1 == num_inner) {
        //     copy_tile_init(cb_mask);
        //     copy_tile(cb_mask, /* tile_idx */ 0, /* register idx */ 1U);

        //     mask_tile_init();
        //     mask_tile(0, 1U);
        // }
        // #endif

        // mul_binary_tile_init();
        // mul_binary_tile(0, 0);

        // add_binary_tile_init();
        // add_binary_tile(accum_dst, 0);
        // }

        // tile_regs_wait();
        // tile_regs_commit();

        // pack_tile(0, cb_rms_intermediate);
        // cb_push_back(cb_rms_intermediate, onetile);

        // tile_regs_release();
        // END DEBUG

        // // reconfig_data_format_srca(cb_rms_intermediate);
        // // pack_reconfig_data_format(cb_rms_intermediate);
        // cb_reserve_back(cb_rms_intermediate, onetile);

        // tile_regs_commit();
        // tile_regs_wait();

        // pack_tile(accum_dst, cb_rms_intermediate);
        // tile_regs_release();
        // cb_push_back(cb_rms_intermediate, onetile);

        // DPRINT << "Processed row " << row << " rms intermediate prepared" << ENDL();

        // // reduce and sqrt
        // {
        //     tile_regs_acquire();
        //     // DPRINT << "Processing row " << row << " acquired regs" << ENDL();
        //     reconfig_data_format(cb_rms_intermediate, cb_scaler);
        //     reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_rms_intermediate, cb_scaler,
        //     cb_rms_intermediate); DPRINT << "Processed row " << row << " after reduce init" << ENDL();
        //     reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_rms_intermediate, cb_scaler, 0, 0, 0);
        //     reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_rms_intermediate);
        //     DPRINT << "Processed row " << row << " after reduce revert" << ENDL();

        //     sqrt_tile_init();
        //     sqrt_tile(0);

        //     DPRINT << "Processed row " << row << " sqrted" << ENDL();

        //     reconfig_data_format_srca(cb_eps);
        //     copy_tile_init(cb_eps);
        //     copy_tile(cb_eps, 0, 1U);

        //     // DPRINT << "Processed row " << row << " after copy" << ENDL();

        //     add_binary_tile_init();
        //     add_binary_tile(0, 1U);

        //     tile_regs_commit();
        //     tile_regs_wait();

        //     // cb_pop_front(cb_rms_intermediate, onetile);
        //     // cb_reserve_back(cb_rms_intermediate, onetile);
        //     // pack_tile(0, cb_rms_intermediate);
        //     tile_regs_release();
        //     // cb_push_back(cb_rms_intermediate, onetile);
        // }

        // DPRINT << "Processed row " << row << " rms intermediate reduced and sqrted" << ENDL();

        // cb_reserve_back(cb_rms_output, onetile);
        // tile_regs_acquire();
        // copy_tile_init(cb_rms_intermediate);
        // copy_tile(cb_rms_intermediate, 0, 0);
        // tile_regs_commit();
        // tile_regs_wait();
        // // pack_reconfig_data_format(cb_rms_output);
        // pack_tile(0, cb_rms_output);
        // tile_regs_release();
        // cb_push_back(cb_rms_output, onetile);

        // DPRINT << "Processed row " << row << " rms output prepared" << ENDL();

        // cb_wait_front(cb_gamma, num_inner);

        // // DPRINT << "Processing row " << row << " gamma captured" << ENDL();

        // for (uint32_t col = 0; col < num_inner; ++col) {
        //     // DPRINT << "Processing row " << row << " col " << col << ENDL();
        //     cb_reserve_back(cb_output, onetile);

        //     tile_regs_acquire();

        //     mul_bcast_cols_init_short(cb_input, cb_gamma);
        //     mul_tiles_bcast_cols(cb_input, cb_gamma, col, col, 0);

        //     #ifdef DO_MASK_W
        //     if (col + 1 == num_inner) {
        //         copy_tile_init(cb_mask);
        //         copy_tile(cb_mask, 0, 2U);

        //         mask_tile_init();
        //         mask_tile(0, 2U);
        //     }
        //     #endif

        //     copy_tile_init(cb_rms_intermediate);
        //     copy_tile(cb_rms_intermediate, 0, 1U);

        //     div_binary_tile_init();
        //     div_binary_tile(0, 1U);

        //     // DPRINT << "Processed row " << row << " col " << col << " gamma applied" << ENDL();

        //     tile_regs_wait();
        //     tile_regs_commit();

        //     // DPRINT << "Processed row " << row << " col " << col << " prepare before pack" << ENDL();

        //     // pack_reconfig_data_format(cb_output);
        //     pack_tile(0, cb_output);

        //     tile_regs_release();

        //     cb_push_back(cb_output, onetile);
        // }

        // cb_pop_front(cb_input, num_inner);
        cb_pop_front(cb_rms_intermediate, onetile);
    }

    // DPRINT << "Processed all rows" << ENDL();

    // cb_pop_front(cb_gamma, num_inner);
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);

#ifdef DO_MASK_W
    cb_pop_front(cb_mask, onetile);
#endif

    DPRINT << "All done" << ENDL();
}

}  // namespace NAMESPACE
