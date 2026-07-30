// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"

// Adds tmp2 + tmp3 and packs to stat_cb.  When AlsoPackToOutput is true,
// also packs the same result to out_dfb (reserve/push handled internally).
template <bool AlsoPackToOutput>
ALWI void add_and_pack_stat(
    uint32_t dfb_tmp2, uint32_t dfb_tmp3, uint32_t dfb_stat, uint32_t dfb_out, DataflowBuffer& dfb_out_obj) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    CircularBuffer cb_tmp2(dfb_tmp2);
    CircularBuffer cb_tmp3(dfb_tmp3);
    CircularBuffer cb_stat(dfb_stat);

    cb_stat.reserve_back(onetile);
    if constexpr (AlsoPackToOutput) {
        dfb_out_obj.reserve_back(onetile);
    }
    cb_tmp2.wait_front(1);
    cb_tmp3.wait_front(1);

    tile_regs_acquire();
    add_tiles_init_with_dt(dfb_tmp2, dfb_tmp3);
    add_tiles(dfb_tmp2, dfb_tmp3, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, dfb_stat);
    if constexpr (AlsoPackToOutput) {
        pack_tile(dst0, dfb_out);
    }
    tile_regs_release();

    cb_tmp2.pop_front(1);
    cb_tmp3.pop_front(1);
    cb_stat.push_back(onetile);
    if constexpr (AlsoPackToOutput) {
        dfb_out_obj.push_back(onetile);
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

    DataflowBuffer dfb_out0_obj(dfb_out0);
    DataflowBuffer dfb_momentum_obj(dfb_momentum);
    DataflowBuffer dfb_one_obj(dfb_one);

    binary_op_init_common(dfb_batch_mean, dfb_batch_var, dfb_out0);
    constexpr uint32_t onetile = 1;

    dfb_one_obj.wait_front(1);
    dfb_momentum_obj.wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat

        if constexpr (old_running_mean_has_value) {
            sub_tiles_to_cb(dfb_one, dfb_momentum, dfb_tmp1, 0, 0, 0, 0);           // 1 - momentum
            mul_tiles_to_cb(dfb_momentum, dfb_batch_mean, dfb_tmp2, 0, 0, 0, 1);    // momentum * batch_mean
            mul_tiles_to_cb(dfb_tmp1, dfb_old_running_mean, dfb_tmp3, 0, 0, 1, 1);  // (1-momentum) * running_mean
            add_and_pack_stat<!old_running_var_has_value>(
                dfb_tmp2, dfb_tmp3, dfb_updated_running_mean, dfb_out0, dfb_out0_obj);
        }

        if constexpr (old_running_var_has_value) {
            sub_tiles_to_cb(dfb_one, dfb_momentum, dfb_tmp1, 0, 0, 0, 0);          // 1 - momentum
            mul_tiles_to_cb(dfb_momentum, dfb_batch_var, dfb_tmp2, 0, 0, 0, 1);    // momentum * batch_var
            mul_tiles_to_cb(dfb_tmp1, dfb_old_running_var, dfb_tmp3, 0, 0, 1, 1);  // (1-momentum) * running_var
            add_and_pack_stat<true>(dfb_tmp2, dfb_tmp3, dfb_updated_running_var, dfb_out0, dfb_out0_obj);
        }

        if constexpr (!old_running_mean_has_value && !old_running_var_has_value) {
            // Neither stat present: copy batch_mean to the output so the writer
            // kernel has a tile to consume.
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

        // Drain batch stat CBs not consumed by the blocks above.
        // Reader pushes batch_mean and writer pushes batch_var every tile
        // unconditionally; when a stat is absent the corresponding mul compiles
        // out, so we must pop here to prevent the CB from filling and stalling
        // the producer.
        if constexpr (!old_running_mean_has_value && old_running_var_has_value) {
            CircularBuffer cb_bm(dfb_batch_mean);
            cb_bm.wait_front(onetile);
            cb_bm.pop_front(onetile);
        }
        if constexpr (!old_running_var_has_value) {
            CircularBuffer cb_bv(dfb_batch_var);
            cb_bv.wait_front(onetile);
            cb_bv.pop_front(onetile);
        }
    }

    dfb_one_obj.pop_front(1);
    dfb_momentum_obj.pop_front(1);
}
