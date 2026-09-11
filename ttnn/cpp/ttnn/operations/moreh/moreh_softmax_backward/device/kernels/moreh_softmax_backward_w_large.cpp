// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef LOG
constexpr auto reduce_input = dfb::dy;
#else
constexpr auto reduce_input = dfb::ydy;
#endif
constexpr uint32_t reduce_call_count = get_compile_time_arg_val(0);
template <uint32_t I>
using ReduceCall = ttnn::kernel_lib::
    BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, I>, reduce_input, dfb::scaler, dfb::sum, dfb::add>;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dy_obj(dfb::dy);
    DataflowBuffer dfb_dx_obj(dfb::dx);

    DataflowBuffer dfb_ydy_obj(dfb::ydy);  // y * dy
    DataflowBuffer dfb_sum_obj(dfb::sum);
    DataflowBuffer dfb_dy_m_sum_obj(dfb::dy_m_sum);

    compute_kernel_hw_startup(dfb::y, dfb::scaler, dfb::dx);

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // The streaming plan reduces dy directly, including its partial tail.
        compute_kernel_lib::reduce<ReduceCall<0>>();

        for (uint32_t w = 0; w < Wt; w += onetile) {
            // exp(y)
            auto& dfb_exp_obj = dfb_ydy_obj;  // the y * dy buffer, reused to hold exp(y)
            exp_tile_to_cb(dfb_y_obj, dfb_exp_obj, 0);
            // sum * exp(y)
            mul_tiles_bcast_cols_to_cb(dfb_exp_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

            // dy - sum * exp(y)
            sub_tiles_to_cb(dfb_dy_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
        }

        dfb_sum_obj.pop_front(onetile);
#else
        constexpr uint32_t block_tiles = get_arg(args::reduce_block_tiles);
        constexpr uint32_t buffer_tiles = get_arg(args::reduce_buffer_tiles);
        const uint32_t num_blocks = Wt < block_tiles ? 1 : Wt / block_tiles;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t current_tiles = block + 1 == num_blocks ? Wt - block * block_tiles : block_tiles;
            dfb_ydy_obj.reserve_back(buffer_tiles);
            for (uint32_t tile = 0; tile < current_tiles; ++tile) {
                dfb_y_obj.wait_front(1);
                dfb_dy_obj.wait_front(1);
                tile_regs_acquire();
                mul_tiles_init_with_dt(dfb_y_obj, dfb_dy_obj);
                mul_tiles(dfb::y, dfb::dy, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(dfb::ydy);
                pack_tile<true>(0, dfb::ydy, tile);
                tile_regs_release();
                dfb_y_obj.pop_front(1);
                dfb_dy_obj.pop_front(1);
            }
            dfb_ydy_obj.push_back(buffer_tiles);
            if (block == 0) {
                compute_kernel_lib::reduce<ReduceCall<0>>();
            } else if constexpr (reduce_call_count > 1) {
                if (block + 1 == num_blocks) {
                    compute_kernel_lib::reduce<ReduceCall<reduce_call_count - 1>>();
                } else {
                    compute_kernel_lib::reduce<ReduceCall<1>>();
                }
            }
            dfb_ydy_obj.pop_front(buffer_tiles);
        }

        // step 3, compute final result
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // dy - sum
            sub_tiles_bcast_cols_to_cb(dfb_dy_obj, dfb_sum_obj, dfb_dy_m_sum_obj, 0, 0, /*pop0=*/1, /*pop1=*/0);

#ifdef SOFTMAX
            // (dy - sum) * y
            mul_tiles_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
#else
            // -(dy - sum) * y
            mul_tiles_and_negative_to_cb(dfb_y_obj, dfb_dy_m_sum_obj, dfb_dx_obj);
#endif
        }

        dfb_sum_obj.pop_front(onetile);
#endif
    }
    DataflowBuffer(dfb::scaler).pop_front(get_arg(args::reduce_auxiliary_tiles));
}
