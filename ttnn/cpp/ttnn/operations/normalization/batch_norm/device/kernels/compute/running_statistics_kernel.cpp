// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"

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

    DataflowBuffer dfb_out0_obj(dfb_out0);
    DataflowBuffer dfb_momentum_obj(dfb_momentum);
    DataflowBuffer dfb_one_obj(dfb_one);

    binary_op_init_common(dfb_batch_mean, dfb_batch_var, dfb_out0);
    constexpr uint32_t onetile = 1;

    dfb_one_obj.wait_front(1);
    dfb_momentum_obj.wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
        //
        // The *_tiles_to_cb helpers each run their own tile_regs_acquire/release cycle, so
        // wrapping them in an outer tile_regs bracket is invalid (nested acquire is UB and
        // the final pack would read stale DST). Instead, inline the last add step to keep
        // DST live for packing to both the stat DFB and the output DFB.

        if constexpr (old_running_mean_has_value) {
            sub_tiles_to_cb(dfb_one, dfb_momentum, dfb_tmp1, 0, 0, 0, 0);           // 1 - momentum
            mul_tiles_to_cb(dfb_momentum, dfb_batch_mean, dfb_tmp2, 0, 0, 0, 1);    // momentum * batch_mean
            mul_tiles_to_cb(dfb_tmp1, dfb_old_running_mean, dfb_tmp3, 0, 0, 1, 1);  // (1-momentum) * running_mean

            // Inline final add: tmp2 + tmp3 → updated_running_mean (and → output if var absent)
            {
                constexpr uint32_t dst0 = 0;
                CircularBuffer cb_tmp2(dfb_tmp2);
                CircularBuffer cb_tmp3(dfb_tmp3);
                CircularBuffer cb_stat(dfb_updated_running_mean);

                cb_stat.reserve_back(onetile);
                cb_tmp2.wait_front(1);
                cb_tmp3.wait_front(1);

                tile_regs_acquire();
                add_tiles_init_with_dt(dfb_tmp2, dfb_tmp3);
                add_tiles(dfb_tmp2, dfb_tmp3, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_updated_running_mean);
                if constexpr (!old_running_var_has_value) {
                    dfb_out0_obj.reserve_back(onetile);
                    pack_tile(dst0, dfb_out0);
                }
                tile_regs_release();

                cb_tmp2.pop_front(1);
                cb_tmp3.pop_front(1);
                cb_stat.push_back(onetile);
                if constexpr (!old_running_var_has_value) {
                    dfb_out0_obj.push_back(onetile);
                }
            }
        }

        if constexpr (old_running_var_has_value) {
            sub_tiles_to_cb(dfb_one, dfb_momentum, dfb_tmp1, 0, 0, 0, 0);          // 1 - momentum
            mul_tiles_to_cb(dfb_momentum, dfb_batch_var, dfb_tmp2, 0, 0, 0, 1);    // momentum * batch_var
            mul_tiles_to_cb(dfb_tmp1, dfb_old_running_var, dfb_tmp3, 0, 0, 1, 1);  // (1-momentum) * running_var

            // Inline final add: tmp2 + tmp3 → updated_running_var and → output
            {
                constexpr uint32_t dst0 = 0;
                CircularBuffer cb_tmp2(dfb_tmp2);
                CircularBuffer cb_tmp3(dfb_tmp3);
                CircularBuffer cb_stat(dfb_updated_running_var);

                cb_stat.reserve_back(onetile);
                dfb_out0_obj.reserve_back(onetile);
                cb_tmp2.wait_front(1);
                cb_tmp3.wait_front(1);

                tile_regs_acquire();
                add_tiles_init_with_dt(dfb_tmp2, dfb_tmp3);
                add_tiles(dfb_tmp2, dfb_tmp3, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, dfb_updated_running_var);
                pack_tile(dst0, dfb_out0);
                tile_regs_release();

                cb_tmp2.pop_front(1);
                cb_tmp3.pop_front(1);
                cb_stat.push_back(onetile);
                dfb_out0_obj.push_back(onetile);
            }
        }

        if constexpr (!old_running_mean_has_value && !old_running_var_has_value) {
            // Neither stat present: copy batch_mean to the output rather than leaving the
            // reserved tile uninitialised. Matches SFPU both-absent handling. The return
            // value of running_statistics is discarded by batch_norm.cpp, but the writer
            // kernel still consumes it.
            CircularBuffer cb_batch_mean(dfb_batch_mean);
            cb_batch_mean.wait_front(onetile);
            dfb_out0_obj.reserve_back(onetile);
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_batch_mean);
            copy_tile(dfb_batch_mean, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_out0);
            tile_regs_release();
            cb_batch_mean.pop_front(onetile);
            dfb_out0_obj.push_back(onetile);
        }
    }

    dfb_one_obj.pop_front(1);
    dfb_momentum_obj.pop_front(1);
}
