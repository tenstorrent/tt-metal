// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"

template <bool NeedsTypecast, uint32_t TcInFmt, uint32_t TcOutFmt>
ALWI void maybe_typecast_stat(
    DataflowBuffer& src_obj, uint32_t src_dfb, uint32_t dst_dfb, uint32_t& last_srca_dfb, uint32_t tile_index) {
    if constexpr (NeedsTypecast) {
        constexpr uint32_t onetile = 1;
        src_obj.wait_front(onetile);
        DataflowBuffer dst_obj(dst_dfb);
        dst_obj.reserve_back(onetile);

        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(last_srca_dfb, src_dfb);
        last_srca_dfb = src_dfb;
        copy_tile(src_dfb, tile_index, tile_index * 2);
        typecast_tile_init<TcInFmt, TcOutFmt>();
        typecast_tile<TcInFmt, TcOutFmt>(tile_index * 2);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(dst_dfb);
        pack_tile(tile_index * 2, dst_dfb);
        tile_regs_release();

        pack_reconfig_data_format(dst_dfb, src_dfb);

        src_obj.pop_front(onetile);
        dst_obj.push_back(onetile);
    }
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(1) == 1;

    constexpr auto dfb_batch_mean = get_compile_time_arg_val(2);  // batch mean
    constexpr auto dfb_batch_var = get_compile_time_arg_val(3);   // batch var
    constexpr auto dfb_out0 = get_compile_time_arg_val(4);
    constexpr auto dfb_old_running_mean = get_compile_time_arg_val(5);      // old running mean tensor
    constexpr auto dfb_old_running_var = get_compile_time_arg_val(6);       // old running var tensor
    constexpr auto dfb_updated_running_mean = get_compile_time_arg_val(7);  // updated running mean tensor
    constexpr auto dfb_updated_running_var = get_compile_time_arg_val(8);   // updated running var tensor
    constexpr auto dfb_momentum = get_compile_time_arg_val(9);              // momentum
    constexpr auto dfb_one = get_compile_time_arg_val(10);                  // stores 1
    constexpr auto dfb_tmp1 = get_compile_time_arg_val(11);                 // tmp 1
    constexpr auto dfb_tmp2 = get_compile_time_arg_val(12);                 // tmp 2
    constexpr auto dfb_tmp3 = get_compile_time_arg_val(13);                 // tmp 3
    constexpr auto dfb_writer_updated_mean = get_compile_time_arg_val(14);  // writer-facing updated mean
    constexpr auto dfb_writer_updated_var = get_compile_time_arg_val(15);   // writer-facing updated var
    constexpr bool stat_needs_typecast = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(17);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(18);
    constexpr bool needs_mean_typecast = old_running_mean_has_value && stat_needs_typecast;
    constexpr bool needs_var_typecast = old_running_var_has_value && stat_needs_typecast;

    DataflowBuffer dfb_batch_mean_obj(dfb_batch_mean);
    DataflowBuffer dfb_batch_var_obj(dfb_batch_var);
    DataflowBuffer dfb_out0_obj(dfb_out0);
    DataflowBuffer dfb_old_running_mean_obj(dfb_old_running_mean);
    DataflowBuffer dfb_old_running_var_obj(dfb_old_running_var);
    DataflowBuffer dfb_updated_running_mean_obj(dfb_updated_running_mean);
    DataflowBuffer dfb_updated_running_var_obj(dfb_updated_running_var);
    DataflowBuffer dfb_momentum_obj(dfb_momentum);
    DataflowBuffer dfb_one_obj(dfb_one);
    DataflowBuffer dfb_tmp1_obj(dfb_tmp1);
    DataflowBuffer dfb_tmp2_obj(dfb_tmp2);
    DataflowBuffer dfb_tmp3_obj(dfb_tmp3);

    unary_op_init_common(dfb_batch_mean, dfb_out0);
    uint32_t last_srca_dfb = dfb_batch_mean;
    constexpr uint32_t onetile = 1;

    dfb_momentum_obj.wait_front(1);
    dfb_one_obj.wait_front(1);

    // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // cb tile index
        constexpr uint32_t tile_index = 0;

        dfb_batch_mean_obj.wait_front(onetile);
        dfb_out0_obj.reserve_back(1);

        if constexpr (old_running_mean_has_value) {
            // 1 - momentum
            dfb_tmp1_obj.reserve_back(onetile);
            tile_regs_acquire();
            sub_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_one);
            last_srca_dfb = dfb_one;
            copy_tile(dfb_one, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_momentum);
            last_srca_dfb = dfb_momentum;
            copy_tile(dfb_momentum, tile_index, tile_index * 2 + 1);
            sub_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(dfb_tmp1);
            pack_tile_with_dt(tile_index * 2, dfb_tmp1);
            tile_regs_release();
            dfb_tmp1_obj.push_back(onetile);

            // momentum * batch stat
            dfb_tmp2_obj.reserve_back(onetile);
            tile_regs_acquire();
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_batch_mean);
            last_srca_dfb = dfb_batch_mean;
            copy_tile(dfb_batch_mean, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_momentum);
            last_srca_dfb = dfb_momentum;
            copy_tile(dfb_momentum, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            // No pack reconfig needed: cb_tmp1 and cb_tmp2 share interm_data_format
            pack_tile_with_dt(tile_index * 2, dfb_tmp2);
            tile_regs_release();
            dfb_tmp2_obj.push_back(onetile);

            // cb_tmp1 * running stats --> (1 - momentum) * running stats
            dfb_tmp1_obj.wait_front(onetile);
            dfb_old_running_mean_obj.wait_front(onetile);
            dfb_tmp3_obj.reserve_back(onetile);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_old_running_mean);
            last_srca_dfb = dfb_old_running_mean;
            copy_tile(dfb_old_running_mean, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp1);
            last_srca_dfb = dfb_tmp1;
            copy_tile(dfb_tmp1, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            // No pack reconfig needed: cb_tmp2 and cb_tmp3 share interm_data_format
            pack_tile_with_dt(tile_index * 2, dfb_tmp3);
            tile_regs_release();
            dfb_tmp3_obj.push_back(onetile);

            dfb_old_running_mean_obj.pop_front(onetile);
            dfb_tmp1_obj.pop_front(onetile);

            // cb_tmp2 + cb_tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            dfb_tmp2_obj.wait_front(onetile);
            dfb_tmp3_obj.wait_front(onetile);
            dfb_updated_running_mean_obj.reserve_back(onetile);
            tile_regs_acquire();
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp3);
            last_srca_dfb = dfb_tmp3;
            copy_tile(dfb_tmp3, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp2);
            last_srca_dfb = dfb_tmp2;
            copy_tile(dfb_tmp2, tile_index, tile_index * 2 + 1);
            add_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            // No pack reconfig needed: cb_tmp3 and cb_updated_running_mean share interm_data_format
            pack_tile_with_dt(tile_index * 2, dfb_updated_running_mean);
            // For the output tensor, return the same values as either of the stats.
            if constexpr (!old_running_var_has_value) {
                pack_reconfig_data_format(dfb_updated_running_mean, dfb_out0);
                pack_tile_with_dt(tile_index * 2, dfb_out0);
            }
            tile_regs_release();
            dfb_updated_running_mean_obj.push_back(onetile);

            maybe_typecast_stat<needs_mean_typecast, tc_in_fmt, tc_out_fmt>(
                dfb_updated_running_mean_obj,
                dfb_updated_running_mean,
                dfb_writer_updated_mean,
                last_srca_dfb,
                tile_index);

            dfb_tmp3_obj.pop_front(onetile);
            dfb_tmp2_obj.pop_front(onetile);
        }

        dfb_batch_mean_obj.pop_front(onetile);

        if constexpr (old_running_var_has_value) {
            // 1 - momentum
            dfb_tmp1_obj.reserve_back(onetile);
            tile_regs_acquire();
            sub_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_one);
            last_srca_dfb = dfb_one;
            copy_tile(dfb_one, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_momentum);
            last_srca_dfb = dfb_momentum;
            copy_tile(dfb_momentum, tile_index, tile_index * 2 + 1);
            sub_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(dfb_tmp1);
            pack_tile_with_dt(tile_index * 2, dfb_tmp1);
            tile_regs_release();
            dfb_tmp1_obj.push_back(onetile);

            // momentum * batch stat
            dfb_batch_var_obj.wait_front(onetile);
            dfb_tmp2_obj.reserve_back(onetile);
            tile_regs_acquire();
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_batch_var);
            last_srca_dfb = dfb_batch_var;
            copy_tile(dfb_batch_var, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_momentum);
            last_srca_dfb = dfb_momentum;
            copy_tile(dfb_momentum, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, dfb_tmp2);
            tile_regs_release();
            dfb_tmp2_obj.push_back(onetile);

            dfb_batch_var_obj.pop_front(onetile);

            // cb_tmp1 * running stats --> (1 - momentum) * running stats
            dfb_tmp1_obj.wait_front(onetile);
            dfb_old_running_var_obj.wait_front(onetile);
            dfb_tmp3_obj.reserve_back(onetile);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_old_running_var);
            last_srca_dfb = dfb_old_running_var;
            copy_tile(dfb_old_running_var, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp1);
            last_srca_dfb = dfb_tmp1;
            copy_tile(dfb_tmp1, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, dfb_tmp3);
            tile_regs_release();
            dfb_tmp3_obj.push_back(onetile);

            dfb_old_running_var_obj.pop_front(onetile);
            dfb_tmp1_obj.pop_front(onetile);

            // cb_tmp2 + cb_tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            dfb_tmp2_obj.wait_front(onetile);
            dfb_tmp3_obj.wait_front(onetile);
            dfb_updated_running_var_obj.reserve_back(onetile);
            tile_regs_acquire();
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp3);
            last_srca_dfb = dfb_tmp3;
            copy_tile(dfb_tmp3, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp2);
            last_srca_dfb = dfb_tmp2;
            copy_tile(dfb_tmp2, tile_index, tile_index * 2 + 1);
            add_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, dfb_updated_running_var);
            pack_reconfig_data_format(dfb_updated_running_var, dfb_out0);
            pack_tile_with_dt(tile_index * 2, dfb_out0);
            tile_regs_release();
            dfb_updated_running_var_obj.push_back(onetile);

            maybe_typecast_stat<needs_var_typecast, tc_in_fmt, tc_out_fmt>(
                dfb_updated_running_var_obj,
                dfb_updated_running_var,
                dfb_writer_updated_var,
                last_srca_dfb,
                tile_index);

            dfb_tmp3_obj.pop_front(onetile);
            dfb_tmp2_obj.pop_front(onetile);
        }

        dfb_out0_obj.push_back(1);
    }
    dfb_momentum_obj.pop_front(1);
    dfb_one_obj.pop_front(1);
}
