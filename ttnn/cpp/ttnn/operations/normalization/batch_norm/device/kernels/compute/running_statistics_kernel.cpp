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
    static_assert(
        old_running_mean_has_value || old_running_var_has_value,
        "running_statistics requires at least one of running_mean / running_var");

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

    DataflowBuffer dfb_batch_mean_obj(dfb_batch_mean);
    DataflowBuffer dfb_batch_var_obj(dfb_batch_var);
    DataflowBuffer dfb_momentum_obj(dfb_momentum);
    DataflowBuffer dfb_one_obj(dfb_one);

    binary_op_init_common(dfb_batch_mean, dfb_batch_var, dfb_out0);
    constexpr uint32_t onetile = 1;

    dfb_one_obj.wait_front(1);
    dfb_momentum_obj.wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
        //
        // HAZARD: reader/writer push batch_mean and batch_var every tile regardless of
        // which stats are present. Both must be waited and popped unconditionally here;
        // omitting a pop will stall the producer after the CB fills (CB depth is 2).

        dfb_batch_mean_obj.wait_front(onetile);
        dfb_batch_var_obj.wait_front(onetile);

        if constexpr (old_running_mean_has_value) {
            sub_tiles_to_cb(dfb_one, dfb_momentum, dfb_tmp1, 0, 0, 0, 0);           // 1 - momentum
            mul_tiles_to_cb(dfb_momentum, dfb_batch_mean, dfb_tmp2, 0, 0, 0, 0);    // momentum * batch_mean
            mul_tiles_to_cb(dfb_tmp1, dfb_old_running_mean, dfb_tmp3, 0, 0, 1, 1);  // (1-momentum) * running_mean
            if constexpr (old_running_var_has_value) {
                // Var block below will pack to dfb_out0.
                add_tiles_to_cb(dfb_tmp2, dfb_tmp3, dfb_updated_running_mean, 0, 0, 1, 1);
            } else {
                // No var block — this is the last compute in the tile, so pack
                // the mean result to both dfb_updated_running_mean and dfb_out0.
                add_tiles_to_two_cbs(dfb_tmp2, dfb_tmp3, dfb_updated_running_mean, dfb_out0, 0, 0, 1, 1);
            }
        }

        if constexpr (old_running_var_has_value) {
            sub_tiles_to_cb(dfb_one, dfb_momentum, dfb_tmp1, 0, 0, 0, 0);          // 1 - momentum
            mul_tiles_to_cb(dfb_momentum, dfb_batch_var, dfb_tmp2, 0, 0, 0, 0);    // momentum * batch_var
            mul_tiles_to_cb(dfb_tmp1, dfb_old_running_var, dfb_tmp3, 0, 0, 1, 1);  // (1-momentum) * running_var
            // Last compute in the tile — pack to both dfb_updated_running_var and dfb_out0.
            add_tiles_to_two_cbs(dfb_tmp2, dfb_tmp3, dfb_updated_running_var, dfb_out0, 0, 0, 1, 1);
        }

        dfb_batch_mean_obj.pop_front(onetile);
        dfb_batch_var_obj.pop_front(onetile);
    }

    dfb_one_obj.pop_front(1);
    dfb_momentum_obj.pop_front(1);
}
