// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_max = tt::CBIndex::c_26;
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    cb_wait_front(cb_mask, onetile);
    cb_wait_front(cb_max_scaler, onetile);
    cb_wait_front(cb_sum_scaler, onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: bulk reduce of Wt-1 full tiles into cb_max (pack preserves full DEST).
            cb_reserve_back(cb_max, 1);

            tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_in0, cb_max_scaler);
#endif
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, cb_max);
            for (uint32_t x = 0; x < Wt - 1; ++x) {
                cb_wait_front(cb_in0, x + 1);
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, x, 0, dst0);
            }
            reduce_uninit();
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_push_back(cb_max, 1);

            // Phase 2: merge the masked last tile into cb_max.
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

            cb_wait_front(cb_max, 1);
            cb_wait_front(cb_tmp, 1);

            tile_regs_acquire();
            copy_tile_init_with_dt(cb_max);
            copy_tile(cb_max, 0, dst0);

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_tmp, cb_max_scaler);
#endif
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tmp, cb_max_scaler, cb_max);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_tmp, cb_max_scaler, 0, 0, dst0);
            reduce_uninit();
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_pop_front(cb_max, 1);
            cb_pop_front(cb_tmp, 1);
            cb_push_back(cb_max, 1);
        }

        // compute x - max(x)
        // PARTIAL migration: A walks Wt tiles upfront, B (cb_max) is col-broadcast
        // pinned tile 0 (already waited). Chain owns cb_in0 / cb_x_m_max lifecycle;
        // caller still owns cb_max pop.
        cb_wait_front(cb_max, 1);
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                Wt,
                BinaryFpu<
                    cb_in0,
                    cb_max,
                    BinaryFpuOp::Sub,
                    BroadcastDim::Col,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitUpfrontPopAtEnd,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::BlockIter,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                PackTile<
                    cb_x_m_max,
                    Dst::D0,
                    PackTilePolicy::UpfrontReservePushAtEnd,
                    PackTileIndexMode::BlockIter,
                    PackTileReconfig::Output>{});
        }
        cb_pop_front(cb_max, 1);

        // compute exp(x - max(x))  — last tile is masked.
        // PARTIAL migration: split into two chains. Tiles [0, Wt-1) plain
        // copy + (negative if SOFTMIN) + exp. Tile Wt-1 additionally loads mask
        // into D1 and runs Mask. cb_x_m_max + cb_mask pre-waited (NoWaitNoPop).
        cb_wait_front(cb_x_m_max, Wt);
        {
            using namespace compute_kernel_lib;
            if (Wt > 1) {
                eltwise_chain(
                    Wt - 1,
                    CopyTile<
                        cb_x_m_max,
                        Dst::D0,
                        CopyTilePolicy::NoWaitNoPop,
                        CbIndexMode::BlockIter,
                        CopyTileReconfig::Input>{},
#ifndef SOFTMAX
                    Negative<Dst::D0>{},
#endif
                    Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
                    PackTile<
                        cb_exps,
                        Dst::D0,
                        PackTilePolicy::PerTileReserveAndPush,
                        PackTileIndexMode::FirstTile,
                        PackTileReconfig::Output>{});
            }
            // Last tile — masked path.
            eltwise_chain(
                1,
                CopyTile<
                    cb_x_m_max,
                    Dst::D0,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::Pinned,
                    CopyTileReconfig::Input>{Wt - 1},
                CopyTile<
                    cb_mask,
                    Dst::D1,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::FirstTile,
                    CopyTileReconfig::Input>{},
#ifndef SOFTMAX
                Negative<Dst::D0>{},
#endif
                Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
                Mask<DataFormat::Float16_b, Dst::D0>{},
                PackTile<
                    cb_exps,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

#ifdef LOG
        // log(sum) - pop tiles after reduce
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_exps,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::row(Wt),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    log_tile_init();
                    log_tile(dst_idx);
                });
#else
        // 1/sum - keep tiles for subsequent multiplication
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_exps,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::row(Wt),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });
#endif

        // compute final result
        // PARTIAL migration: final divide stage as eltwise_chain.
        // LOG branch  : out = (x - max) - log(sum)   — sub with cb_recipsumexps col-bcast
        // !LOG branch : out = exp(x - max) / sum     — mul with cb_recipsumexps col-bcast
        // A walks Wt tiles (BlockIter), B (cb_recipsumexps) col-broadcast pinned
        // at tile 0 (NoWaitNoPop). Chain owns per-tile pack reserve/push on cb_out0.
        cb_wait_front(cb_x_m_max, Wt);
        cb_wait_front(cb_recipsumexps, 1);
#ifndef LOG
        cb_wait_front(cb_exps, Wt);
#endif

        {
            using namespace compute_kernel_lib;
#ifdef LOG
            eltwise_chain(
                Wt,
                BinaryFpu<
                    cb_x_m_max,
                    cb_recipsumexps,
                    BinaryFpuOp::Sub,
                    BroadcastDim::Col,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::NoWaitNoPop,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::BlockIter,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                PackTile<
                    cb_out0,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
#else
            eltwise_chain(
                Wt,
                BinaryFpu<
                    cb_exps,
                    cb_recipsumexps,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Col,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::NoWaitNoPop,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::BlockIter,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                PackTile<
                    cb_out0,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
#endif
        }

        cb_pop_front(cb_recipsumexps, 1);
        cb_pop_front(cb_x_m_max, Wt);
#ifndef LOG
        cb_pop_front(cb_exps, Wt);
#endif
    }
}
