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

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_in0, cb_bcast_scaler, cb_out0);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    cb_wait_front(cb_mask, onetile);
    cb_wait_front(cb_bcast_scaler, onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Ht == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL>(
                cb_tmp, cb_bcast_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            compute_kernel_lib::
                reduce<PoolType::MAX, ReduceDim::REDUCE_COL, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                    cb_in0, cb_bcast_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Ht - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL>(
                cb_tmp,
                cb_bcast_scaler,
                cb_max,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, 1));  // iteration=1, reload from cb_max
        }

        // compute x - max(x)
        cb_reserve_back(cb_x_m_max, Ht);
        cb_wait_front(cb_in0, Ht);
        cb_wait_front(cb_max, 1);

        for (uint32_t h = 0; h < Ht; ++h) {
            tile_regs_acquire();
            sub_bcast_rows_init_short_with_dt(cb_in0, cb_max);
            sub_tiles_bcast<BroadcastType::ROW>(cb_in0, cb_max, h, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_x_m_max);
            tile_regs_release();
        }
        cb_pop_front(cb_max, 1);
        cb_pop_front(cb_in0, Ht);
        cb_push_back(cb_x_m_max, Ht);

        // compute exp(x - max(x))
        cb_reserve_back(cb_exps, Ht);
        cb_wait_front(cb_x_m_max, Ht);
        for (uint32_t h = 0; h < Ht; ++h) {
            tile_regs_acquire();
            copy_tile_init_with_dt(cb_x_m_max);
            copy_tile(cb_x_m_max, h, dst0);

#ifndef SOFTMAX
            negative_tile_init();
            negative_tile(dst0);
#endif

            exp_tile_init();
            exp_tile(dst0);

            if (h == Ht - 1) {
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
        cb_push_back(cb_exps, Ht);

#ifdef LOG
        // log(sum) - pop tiles after reduce
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_COL, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerBatch>(
                cb_exps,
                cb_bcast_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::col(Ht),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    log_tile_init();
                    log_tile(dst_idx);
                });
#else
        // 1/sum - keep tiles for subsequent multiplication
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_COL, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_exps,
                cb_bcast_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::col(Ht),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });
#endif

        // compute final result
        cb_reserve_back(cb_out0, Ht);
        cb_wait_front(cb_x_m_max, Ht);
        cb_wait_front(cb_recipsumexps, 1);
#ifndef LOG
        cb_wait_front(cb_exps, Ht);
#endif

        for (uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
            // x - max - log(sum)
            tile_regs_acquire();
            sub_bcast_rows_init_short_with_dt(cb_x_m_max, cb_recipsumexps);
            sub_tiles_bcast<BroadcastType::ROW>(cb_x_m_max, cb_recipsumexps, h, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            tile_regs_release();
#else
            // exp(x - max) / psum
            tile_regs_acquire();
            mul_bcast_rows_init_short_with_dt(cb_exps, cb_recipsumexps);
            mul_tiles_bcast_rows(cb_exps, cb_recipsumexps, h, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_out0);
            tile_regs_release();
#endif
        }

        cb_pop_front(cb_recipsumexps, 1);
        cb_pop_front(cb_x_m_max, Ht);
        cb_push_back(cb_out0, Ht);
#ifndef LOG
        cb_pop_front(cb_exps, Ht);
#endif
    }
}
