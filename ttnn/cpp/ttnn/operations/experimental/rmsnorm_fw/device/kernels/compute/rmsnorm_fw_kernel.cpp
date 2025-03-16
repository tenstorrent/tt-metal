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
#include "compute_kernel_api/eltwise_unary/recip.h"
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
    constexpr auto cb_output_intermediate = tt::CBIndex::c_8;

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);

#ifdef DO_MASK_W
    cb_wait_front(cb_mask, onetile);
#endif

    init_sfpu(cb_input, cb_output);
    binary_op_init_common(cb_input, cb_gamma, cb_output);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_input, num_inner);
        cb_reserve_back(cb_rms_intermediate, onetile);

        const uint32_t accum_register = 0;
        const uint32_t tile_register = 1U;
        const uint32_t mask_register = 2U;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; ++col) {
            auto working_register = col == 0 ? accum_register : tile_register;
            copy_tile_init(cb_input);
            copy_tile(cb_input, /* tile_idx */ col, /* register_idx */ working_register);

#ifdef DO_MASK_W
            if (col + 1 == num_inner) {
                copy_tile_init(cb_mask);
                copy_tile(cb_mask, /* tile_idx */ 0, /* register idx */ mask_register);

                mask_tile_init();
                mask_tile(working_register, mask_register);
            }
#endif

            mul_binary_tile_init();
            mul_binary_tile(working_register, working_register);

            if (col != 0) {
                add_binary_tile_init();
                add_binary_tile(accum_register, working_register);
            }
        }
        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_rms_intermediate);
        pack_tile(accum_register, cb_rms_intermediate);
        cb_push_back(cb_rms_intermediate, onetile);
        tile_regs_release();

        // reduce and sqrt
        {
            cb_wait_front(cb_rms_intermediate, onetile);
            cb_reserve_back(cb_rms_intermediate, onetile);
            tile_regs_acquire();

            const uint32_t reduction_register = 0;
            reconfig_data_format(cb_rms_intermediate, cb_scaler);
            reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_rms_intermediate, cb_scaler, cb_rms_intermediate);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_rms_intermediate, cb_scaler, /* tile_idx */ 0, /* tile_idx */ 0, reduction_register);
            reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_rms_intermediate);

            const uint32_t eps_register = 1U;
            reconfig_data_format_srca(cb_eps);
            copy_tile_init(cb_eps);
            copy_tile(cb_eps, /* tile_idx */ 0, /* register_idx */ eps_register);

            reconfig_data_format(cb_rms_intermediate, cb_eps);
            add_binary_tile_init();
            add_binary_tile(reduction_register, eps_register);

            sqrt_tile_init();
            sqrt_tile(reduction_register);

            tile_regs_commit();
            tile_regs_wait();

            pack_reconfig_data_format(cb_rms_intermediate);
            pack_tile(reduction_register, cb_rms_intermediate);

            tile_regs_release();
            cb_push_back(cb_rms_intermediate, onetile);
        }

        // copy tile from cb_rms_intermediate to cb_rms_output
        {
            const uint32_t temporary_register = 0;

            cb_wait_front(cb_rms_intermediate, /* num tiles */ 2U);
            tile_regs_acquire();
            cb_reserve_back(cb_rms_output, onetile);
            copy_tile_init(cb_rms_intermediate);
            copy_tile(cb_rms_intermediate, /* tile idx */ 1U, temporary_register);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(cb_rms_output);
            pack_tile(temporary_register, cb_rms_output);
            tile_regs_release();
            cb_push_back(cb_rms_output, onetile);
        }

        // reciprocal of rms intermediate
        {
            const uint32_t temporary_register = 0;
            cb_wait_front(cb_rms_intermediate, /* num_tiles */ 2U);
            cb_reserve_back(cb_rms_intermediate, onetile);
            tile_regs_acquire();

            copy_tile_init(cb_rms_intermediate);
            copy_tile(cb_rms_intermediate, /* tile idx */ 1U, temporary_register);
            recip_tile_init();
            recip_tile(temporary_register);

            tile_regs_wait();
            tile_regs_commit();

            pack_reconfig_data_format(cb_rms_intermediate);
            pack_tile(temporary_register, cb_rms_intermediate);

            tile_regs_release();
            cb_push_back(cb_rms_intermediate, onetile);
        }

        cb_wait_front(cb_gamma, num_inner);
        cb_wait_front(cb_rms_intermediate, 3U);
        for (uint32_t col = 0; col < num_inner; ++col) {
            cb_reserve_back(cb_output, onetile);
            cb_reserve_back(cb_output_intermediate, onetile);

            tile_regs_acquire();
            reconfig_data_format(cb_input, cb_gamma);
            mul_bcast_rows_init_short(cb_input, cb_gamma);
            mul_tiles_bcast_rows(cb_input, cb_gamma, col, col, 0);

#ifdef DO_MASK_W
            if (col + 1 == num_inner) {
                copy_tile_init(cb_mask);
                copy_tile(cb_mask, 0, 2U);

                mask_tile_init();
                mask_tile(0, 2U);
            }
#endif

            tile_regs_wait();
            tile_regs_commit();
            pack_reconfig_data_format(cb_output_intermediate);
            pack_tile(0, cb_output_intermediate);
            cb_push_back(cb_output_intermediate, onetile);
            tile_regs_release();

            cb_wait_front(cb_output_intermediate, onetile);
            tile_regs_acquire();
            reconfig_data_format(cb_output_intermediate, cb_rms_intermediate);
            mul_bcast_cols_init_short(cb_output_intermediate, cb_rms_intermediate);
            mul_tiles_bcast_cols(cb_output_intermediate, cb_rms_intermediate, 0, 2U, 0);

            tile_regs_wait();
            tile_regs_commit();

            pack_reconfig_data_format(cb_output);
            pack_tile(0, cb_output);

            tile_regs_release();
            cb_push_back(cb_output, onetile);
            cb_pop_front(cb_output_intermediate, onetile);
        }

        cb_pop_front(cb_input, num_inner);
        cb_pop_front(cb_rms_intermediate, /* num tiles */ 3U);
    }

    cb_pop_front(cb_gamma, num_inner);
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);

#ifdef DO_MASK_W
    cb_pop_front(cb_mask, onetile);
#endif

    DPRINT << "All done" << ENDL();
}

}  // namespace NAMESPACE
