// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_max = tt::CBIndex::c_26;
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    cb_wait_front(cb_mask, onetile);
    cb_wait_front(cb_bcast_scaler, onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp, cb_bcast_scaler, cb_max, compute_kernel_lib::InputBlockShape::single());
        } else {
            compute_kernel_lib::
                reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, compute_kernel_lib::reduce_policies::PersistentPolicy>(
                    cb_in0, cb_bcast_scaler, cb_max, compute_kernel_lib::InputBlockShape::row(Wt - 1));

            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp,
                cb_bcast_scaler,
                cb_max,
                compute_kernel_lib::InputBlockShape::single(),
                compute_kernel_lib::InputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, 1));  // iteration=1, reload from cb_max
        }

        // compute x - max(x)
        cb_reserve_back(cb_x_m_max, Wt);
        cb_wait_front(cb_in0, Wt);
        cb_wait_front(cb_max, 1);

        for (uint32_t w = 0; w < Wt; ++w) {
            tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(cb_in0, cb_max);
            sub_tiles_bcast<BroadcastType::COL>(cb_in0, cb_max, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_x_m_max);
            tile_regs_release();
        }
        cb_pop_front(cb_max, 1);
        cb_pop_front(cb_in0, Wt);
        cb_push_back(cb_x_m_max, Wt);

        // compute exp(x - max(x))
        cb_reserve_back(cb_exps, Wt);
        cb_wait_front(cb_x_m_max, Wt);
        for (uint32_t w = 0; w < Wt; ++w) {
            tile_regs_acquire();
            copy_tile_init_with_dt(cb_x_m_max);
            copy_tile(cb_x_m_max, w, dst0);

#ifndef SOFTMAX
            negative_tile_init();
            negative_tile(dst0);
#endif

            exp_tile_init();
            exp_tile(dst0);

            if (w == Wt - 1) {
                copy_tile_init_with_dt(cb_mask);
                copy_tile(cb_mask, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_exps);
            tile_regs_release();
        }
        cb_push_back(cb_exps, Wt);

#ifdef LOG
        // log(sum) - pop tiles after reduce
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::reduce_policies::StreamingBatchedPolicy>(
                cb_exps,
                cb_bcast_scaler,
                cb_recipsumexps,
                compute_kernel_lib::InputBlockShape::row(Wt),
                compute_kernel_lib::InputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    log_tile_init();
                    log_tile(dst_idx);
                });
#else
        // 1/sum - keep tiles for subsequent multiplication
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::reduce_policies::PersistentPolicy>(
                cb_exps,
                cb_bcast_scaler,
                cb_recipsumexps,
                compute_kernel_lib::InputBlockShape::row(Wt),
                compute_kernel_lib::InputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });
#endif

        // compute final result
        cb_reserve_back(cb_out0, Wt);
        cb_wait_front(cb_x_m_max, Wt);
        cb_wait_front(cb_recipsumexps, 1);

#ifndef LOG
        cb_wait_front(cb_exps, Wt);
#endif

        for (uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
            // x - max - log(sum)
            tile_regs_acquire();
            sub_bcast_cols_init_short_with_dt(cb_x_m_max, cb_recipsumexps);
            sub_tiles_bcast<BroadcastType::COL>(cb_x_m_max, cb_recipsumexps, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            tile_regs_release();
#else
            // exp(x - max) / psum
            tile_regs_acquire();
            mul_bcast_cols_init_short_with_dt(cb_exps, cb_recipsumexps);
            mul_tiles_bcast_cols(cb_exps, cb_recipsumexps, w, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            tile_regs_release();
#endif
        }

        cb_pop_front(cb_recipsumexps, 1);
        cb_pop_front(cb_x_m_max, Wt);
        cb_push_back(cb_out0, Wt);
#ifndef LOG
        cb_pop_front(cb_exps, Wt);
#endif
    }
}
