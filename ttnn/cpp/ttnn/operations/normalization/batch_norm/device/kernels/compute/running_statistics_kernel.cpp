// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t old_running_mean_has_value = get_arg(args::old_running_mean_has_value) == 1;
    constexpr uint32_t old_running_var_has_value = get_arg(args::old_running_var_has_value) == 1;
    static_assert(
        old_running_mean_has_value || old_running_var_has_value,
        "running_statistics requires at least one of running_mean / running_var");

    DataflowBuffer dfb_batch_mean_obj(dfb::batch_mean);
    DataflowBuffer dfb_batch_var_obj(dfb::batch_var);
    DataflowBuffer dfb_momentum_obj(dfb::momentum);
    DataflowBuffer dfb_one_obj(dfb::one);  // holds 1, for the (1 - momentum) term

    binary_op_init_common(dfb::batch_mean, dfb::batch_var, dfb::out);
    constexpr uint32_t onetile = 1;

    dfb_one_obj.wait_front(1);
    dfb_momentum_obj.wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
        //
        // HAZARD: reader/writer push batch_mean and batch_var every tile regardless of
        // which stats are present. Both must be waited and popped unconditionally here;
        // omitting a pop will stall the producer after the DFB fills (DFB depth is 2).

        dfb_batch_mean_obj.wait_front(onetile);
        dfb_batch_var_obj.wait_front(onetile);

        if constexpr (old_running_mean_has_value) {
            sub_tiles_to_cb(dfb::one, dfb::momentum, dfb::tmp1, 0, 0, 0, 0);           // 1 - momentum
            mul_tiles_to_cb(dfb::momentum, dfb::batch_mean, dfb::tmp2, 0, 0, 0, 0);    // momentum * batch_mean
            mul_tiles_to_cb(dfb::tmp1, dfb::old_running_mean, dfb::tmp3, 0, 0, 1, 1);  // (1-momentum) * running_mean
            if constexpr (old_running_var_has_value) {
                // Var block below will pack to dfb::out.
                add_tiles_to_cb(dfb::tmp2, dfb::tmp3, dfb::updated_mean, 0, 0, 1, 1);
            } else {
                // No var block — this is the last compute in the tile, so pack
                // the mean result to both dfb::updated_mean and dfb::out.
                add_tiles_to_two_cbs(dfb::tmp2, dfb::tmp3, dfb::updated_mean, dfb::out, 0, 0, 1, 1);
            }
        }

        if constexpr (old_running_var_has_value) {
            sub_tiles_to_cb(dfb::one, dfb::momentum, dfb::tmp1, 0, 0, 0, 0);          // 1 - momentum
            mul_tiles_to_cb(dfb::momentum, dfb::batch_var, dfb::tmp2, 0, 0, 0, 0);    // momentum * batch_var
            mul_tiles_to_cb(dfb::tmp1, dfb::old_running_var, dfb::tmp3, 0, 0, 1, 1);  // (1-momentum) * running_var
            // Last compute in the tile — pack to both dfb::updated_var and dfb::out.
            add_tiles_to_two_cbs(dfb::tmp2, dfb::tmp3, dfb::updated_var, dfb::out, 0, 0, 1, 1);
        }

        dfb_batch_mean_obj.pop_front(onetile);
        dfb_batch_var_obj.pop_front(onetile);
    }

    dfb_one_obj.pop_front(1);
    dfb_momentum_obj.pop_front(1);
}
