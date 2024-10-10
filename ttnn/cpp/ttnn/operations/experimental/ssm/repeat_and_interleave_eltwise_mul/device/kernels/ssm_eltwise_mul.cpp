// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {
void MAIN {
    uint32_t in1_num_blocks = get_arg_val<uint32_t>(0);
    uint32_t in1_num_blocks_h = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0_transposed = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in1_transposed = get_compile_time_arg_val(4);
    constexpr uint32_t cb_in1_bcast_row = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_transposed = get_compile_time_arg_val(6);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t num_rows_in_one_tile = 32;

#ifdef REPEAT_INTERLEAVE_IN1
    binary_op_init_common(cb_in0_transposed, cb_in1_bcast_row);  // TODO: Is there a specific one for bcast mul?
#else
    binary_op_init_common(cb_id_in0, cb_id_in1);
#endif

for(uint32_t block_h_id = 0; block_h_id < in1_num_blocks_h; block_h_id++){

#ifdef REPEAT_IN0
    // Transpose in0
    cb_wait_front(cb_id_in0, onetile);
    // No need to transpose in0 if in1 is not repeat_interleaved
    #ifdef REPEAT_INTERLEAVE_IN1
        tile_regs_acquire();
        tile_regs_wait();

        transpose_wh_init_short(cb_id_in0);
        reconfig_data_format_srca(cb_out_transposed, cb_id_in0);
        pack_reconfig_data_format(cb_id_out, cb_in0_transposed);
        transpose_wh_tile(cb_id_in0, 0, 0);

        cb_reserve_back(cb_in0_transposed, onetile);
        pack_tile(0, cb_in0_transposed);

        tile_regs_commit();
        tile_regs_release();
        cb_push_back(cb_in0_transposed, onetile);
        cb_pop_front(cb_id_in0, onetile);

        cb_wait_front(cb_in0_transposed, onetile);
    #endif
#endif

    for (uint32_t in1_block = 0; in1_block < in1_num_blocks; in1_block++) {
        // Transpose in1
        cb_wait_front(cb_id_in1, onetile);
        tile_regs_acquire();
        tile_regs_wait();

// If input b is not repeat_interleaved, then no need to transpose, bcast row
#ifndef REPEAT_INTERLEAVE_IN1
        mul_tiles_init(cb_id_in0, cb_id_in1);
        reconfig_data_format_srca(cb_id_out, cb_id_in0);
        pack_reconfig_data_format(cb_in0_transposed, cb_id_out);
        mul_tiles(cb_id_in0, cb_id_in1, 0, 0, 0);

        cb_reserve_back(cb_id_out, onetile);
        pack_tile(0, cb_id_out);

        tile_regs_commit();
        tile_regs_release();
        cb_push_back(cb_id_out, onetile);
        cb_pop_front(cb_id_in1, onetile);
#else
        transpose_wh_init_short(cb_id_in1);
        reconfig_data_format_srca(cb_id_in1);
        pack_reconfig_data_format(cb_in1_transposed);
        transpose_wh_tile(cb_id_in1, 0, 0);

        cb_reserve_back(cb_in1_transposed, onetile);
        pack_tile(0, cb_in1_transposed);

        tile_regs_commit();
        tile_regs_release();
        cb_push_back(cb_in1_transposed, onetile);
        cb_pop_front(cb_id_in1, onetile);

        // Receive in1 as single rows to bcast mul with in0
        for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
#ifndef REPEAT_IN0
            // Transpose in0
            cb_wait_front(cb_id_in0, onetile);
            tile_regs_acquire();
            tile_regs_wait();

            transpose_wh_init_short(cb_id_in0);
            reconfig_data_format_srca(cb_id_in0);
            pack_reconfig_data_format(cb_in0_transposed);
            transpose_wh_tile(cb_id_in0, 0, 0);

            cb_reserve_back(cb_in0_transposed, onetile);
            pack_tile(0, cb_in0_transposed);

            tile_regs_commit();
            tile_regs_release();
            cb_push_back(cb_in0_transposed, onetile);
            cb_pop_front(cb_id_in0, onetile);

            cb_wait_front(cb_in0_transposed, onetile);
#endif

            cb_wait_front(cb_in1_bcast_row, onetile);
            tile_regs_acquire();
            tile_regs_wait();

            mul_bcast_rows_init_short(cb_in0_transposed, cb_in1_bcast_row);
            reconfig_data_format_srca(cb_in0_transposed);
            pack_reconfig_data_format(cb_out_transposed);
            mul_tiles_bcast_rows(cb_in0_transposed, cb_in1_bcast_row, 0, 0, 0);

            cb_reserve_back(cb_out_transposed, onetile);
            pack_tile(0, cb_out_transposed);

            tile_regs_commit();
            tile_regs_release();
            cb_push_back(cb_out_transposed, onetile);
#ifndef REPEAT_IN0
            cb_pop_front(cb_in0_transposed, onetile);
#endif
            cb_pop_front(cb_in1_bcast_row, onetile);

            // Transpose output back
            cb_wait_front(cb_out_transposed, onetile);
            tile_regs_acquire();
            tile_regs_wait();

            transpose_wh_init_short(cb_out_transposed);
            reconfig_data_format(cb_in0_transposed, cb_out_transposed);
            pack_reconfig_data_format(cb_out_transposed, cb_id_out);
            transpose_wh_tile(cb_out_transposed, 0, 0);

            cb_reserve_back(cb_id_out, onetile);
            pack_tile(0, cb_id_out);

            tile_regs_commit();
            tile_regs_release();
            cb_push_back(cb_id_out, onetile);
            cb_pop_front(cb_out_transposed, onetile);

            /* TODO: Transpose directly on tiles in DST; is something like this possible?
            cb_reserve_back(cb_id_out, onetile);

            tile_regs_acquire();
            tile_regs_wait();
            mul_bcast_rows_init_short(cb_in0_transposed, cb_in1_bcast_row);
            mul_tiles_bcast_rows(cb_in0_transposed, cb_in1_bcast_row, 0, 0, 0);

            MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(true, true, cb_id_out)
            )); MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(0) ));

            pack_tile(0, cb_id_out);

            tile_regs_commit();
            tile_regs_release();
            cb_push_back(cb_id_out, onetile);
            cb_pop_front(cb_in1_bcast_row, onetile);
            */
        }

        cb_pop_front(cb_in1_transposed, onetile);
#endif
    }
#ifdef REPEAT_IN0
#ifdef REPEAT_INTERLEAVE_IN1
    cb_pop_front(cb_in0_transposed, onetile);
#else
    cb_pop_front(cb_id_in0, onetile);
#endif
#endif
}
}
}  // namespace NAMESPACE
