// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t sum_count_offset = ttnn::kernel_lib::reduce_plan_args::call_compile_time_arg_count();
constexpr uint32_t sum_call_count = get_compile_time_arg_val(sum_count_offset);
using MaxCall =
    ttnn::kernel_lib::BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallArgs<0>, dfb::in0, dfb::max_scaler, dfb::max>;
template <uint32_t I>
using SumCall = ttnn::kernel_lib::BoundReduceCallArgs<
    ttnn::kernel_lib::ReduceCallAtT<sum_count_offset + 1, I>,
    dfb::exps,
    dfb::sum_scaler,
    dfb::recip_sum_exps,
    dfb::add>;

void kernel_main() {
    constexpr auto dfb_in0 = dfb::in0;
    DataflowBuffer dfb_in0_obj(dfb_in0);
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    constexpr auto dfb_out0 = dfb::out0;
    DataflowBuffer dfb_out0_obj(dfb_out0);
    constexpr auto dfb_exps = dfb::exps;
    DataflowBuffer dfb_exps_obj(dfb_exps);
    constexpr auto dfb_recipsumexps = dfb::recip_sum_exps;
    DataflowBuffer dfb_recipsumexps_obj(dfb_recipsumexps);
    constexpr auto dfb_add = dfb::add;
    DataflowBuffer dfb_add_obj(dfb_add);
    constexpr auto dfb_max = dfb::max;
    DataflowBuffer dfb_max_obj(dfb_max);
    constexpr auto dfb_tmp = dfb::tmp;
    DataflowBuffer dfb_tmp_obj(dfb_tmp);

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_out0);

    constexpr std::uint32_t onetile = 1;
    constexpr int dst0 = 0;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing and avoid
    // force-unrolling the per-Ht loops (see moreh_softmax_w_large.cpp for the LTO/addrmod rationale).
    std::uint32_t N = get_arg(args::N);
    std::uint32_t Ht = get_arg(args::Ht);

    constexpr uint32_t block_tiles = get_arg(args::reduce_block_tiles);
    constexpr uint32_t buffer_tiles = get_arg(args::reduce_buffer_tiles);
    const uint32_t num_blocks = Ht < block_tiles ? 1 : Ht / block_tiles;
    const auto post_reduce = [](uint32_t dst_idx) {
#ifdef LOG
        log_tile_init();
        log_tile(dst_idx);
#else
        recip_tile_init();
        recip_tile(dst_idx);
#endif
    };

    for (std::uint32_t n = 0; n < N; ++n) {
        compute_kernel_lib::reduce<MaxCall>();

        // Produce bounded resident blocks; the plan masks the last logical
        // tile and performs the tile accumulation plus the within-tile fold.
        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t current_tiles = block + 1 == num_blocks ? Ht - block * block_tiles : block_tiles;
            dfb_exps_obj.reserve_back(buffer_tiles);
            for (uint32_t tile = 0; tile < current_tiles; ++tile) {
                sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
                dfb_tmp_obj.wait_front(1);
                tile_regs_acquire();
                copy_tile_init_with_dt(dfb_tmp_obj);
                copy_tile(dfb_tmp, 0, 0);
#ifndef SOFTMAX
                negative_tile_init();
                negative_tile(0);
#endif
                exp_tile_init();
                exp_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(dfb_exps);
                pack_tile<true>(0, dfb_exps, tile);
                tile_regs_release();
                dfb_tmp_obj.pop_front(1);
            }
            // Consume the full allocation to keep indexed reads aligned.
            dfb_exps_obj.push_back(buffer_tiles);
            if (block == 0) {
                compute_kernel_lib::reduce<SumCall<0>>(post_reduce);
            } else if constexpr (sum_call_count > 1) {
                if (block + 1 == num_blocks) {
                    compute_kernel_lib::reduce<SumCall<sum_call_count - 1>>(post_reduce);
                } else {
                    compute_kernel_lib::reduce<SumCall<1>>(post_reduce);
                }
            }
            dfb_exps_obj.pop_front(buffer_tiles);
        }

        // step 3, compute final result
        for (std::uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            sub_tiles_bcast_rows_to_cb(dfb_tmp_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // -x + max - log(sum)
            // logsoftmin not implemented
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            exp_tile_to_cb(dfb_tmp_obj, dfb_add_obj);

            mul_tiles_bcast_rows_to_cb(dfb_add_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#else
            // rexp(x - max) / sum
            sub_tiles_bcast_rows_to_cb(dfb_in0_obj, dfb_max_obj, dfb_tmp_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            rexp_tile_to_cb(dfb_tmp_obj, dfb_add_obj);

            mul_tiles_bcast_rows_to_cb(dfb_add_obj, dfb_recipsumexps_obj, dfb_out0_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);
#endif
#endif
        }

        dfb_recipsumexps_obj.pop_front(onetile);
        dfb_max_obj.pop_front(onetile);
    }
    DataflowBuffer(dfb_max_scaler).pop_front(MaxCall::auxiliary_tile_count);
    DataflowBuffer(dfb_sum_scaler).pop_front(get_arg(args::sum_auxiliary_tiles));
}
