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
    static_assert(
        !is_null_binding(dfb::old_running_mean) || !is_null_binding(dfb::old_running_var),
        "running_statistics requires at least one of running_mean / running_var");

    DataflowBuffer dfb_batch_mean_obj(dfb::batch_mean);
    DataflowBuffer dfb_batch_var_obj(dfb::batch_var);
    DataflowBuffer dfb_momentum_obj(dfb::momentum);
    DataflowBuffer dfb_one_obj(dfb::one);  // holds 1, for the (1 - momentum) term

    compute_kernel_hw_startup(dfb::batch_mean, dfb::batch_var, dfb::out);
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

        with_nullable_token(dfb::old_running_mean, [&](const DFBBindingToken& old_mean) {
            sub_tiles_to_cb(dfb::one, dfb::momentum, dfb::tmp1, 0, 0, 0, 0);         // 1 - momentum
            mul_tiles_to_cb(dfb::momentum, dfb::batch_mean, dfb::tmp2, 0, 0, 0, 0);  // momentum * batch_mean
            mul_tiles_to_cb(dfb::tmp1, old_mean, dfb::tmp3, 0, 0, 1, 1);             // (1-momentum) * running_mean
            with_nullable_token(dfb::updated_mean, [&](const DFBBindingToken& updated_mean) {
                // Default: no var block, so this is the last compute in the tile and the mean
                // result packs to both updated_mean and out. Presence of var overrides that.
                bool pack_mean_to_out = true;
                with_nullable_token(dfb::old_running_var, [&](const DFBBindingToken&) {
                    add_tiles_to_cb(dfb::tmp2, dfb::tmp3, updated_mean, 0, 0, 1, 1);
                    pack_mean_to_out = false;
                });
                if (pack_mean_to_out) {
                    add_tiles_to_two_cbs(dfb::tmp2, dfb::tmp3, updated_mean, dfb::out, 0, 0, 1, 1);
                }
            });
        });

        with_nullable_token(dfb::old_running_var, [&](const DFBBindingToken& old_var) {
            sub_tiles_to_cb(dfb::one, dfb::momentum, dfb::tmp1, 0, 0, 0, 0);        // 1 - momentum
            mul_tiles_to_cb(dfb::momentum, dfb::batch_var, dfb::tmp2, 0, 0, 0, 0);  // momentum * batch_var
            mul_tiles_to_cb(dfb::tmp1, old_var, dfb::tmp3, 0, 0, 1, 1);             // (1-momentum) * running_var
            with_nullable_token(dfb::updated_var, [&](const DFBBindingToken& updated_var) {
                // Last compute in the tile — pack to both updated_var and out.
                add_tiles_to_two_cbs(dfb::tmp2, dfb::tmp3, updated_var, dfb::out, 0, 0, 1, 1);
            });
        });

        dfb_batch_mean_obj.pop_front(onetile);
        dfb_batch_var_obj.pop_front(onetile);
    }

    dfb_one_obj.pop_front(1);
    dfb_momentum_obj.pop_front(1);
}
