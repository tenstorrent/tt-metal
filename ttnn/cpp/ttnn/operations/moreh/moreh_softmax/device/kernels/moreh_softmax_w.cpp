// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // sub
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    CircularBuffer cb_in0_obj(cb_in0);
    constexpr auto cb_mask = tt::CBIndex::c_1;
    CircularBuffer cb_mask_obj(cb_mask);
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    CircularBuffer cb_max_scaler_obj(cb_max_scaler);
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    CircularBuffer cb_sum_scaler_obj(cb_sum_scaler);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer cb_out0_obj(cb_out0);
    constexpr auto cb_exps = tt::CBIndex::c_24;
    CircularBuffer cb_exps_obj(cb_exps);
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_max = tt::CBIndex::c_26;
    CircularBuffer cb_max_obj(cb_max);
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    CircularBuffer cb_x_m_max_obj(cb_x_m_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;
    CircularBuffer cb_tmp_obj(cb_tmp);

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    cb_mask_obj.wait_front(onetile);
    cb_max_scaler_obj.wait_front(onetile);
    cb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: bulk reduce of Wt-1 full tiles into cb_max (pack preserves full DEST).
            cb_max_obj.reserve_back(1);

            tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_in0, cb_max_scaler);
#endif
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, cb_max);
            for (uint32_t x = 0; x < Wt - 1; ++x) {
                cb_in0_obj.wait_front(x + 1);
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, x, 0, dst0);
            }
            reduce_uninit();
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_max_obj.push_back(1);

            // Phase 2: merge the masked last tile into cb_max.
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Wt - 1, 0, /*pop0=*/0, /*popm=*/0);

            cb_max_obj.wait_front(1);
            cb_tmp_obj.wait_front(1);

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

            cb_max_obj.pop_front(1);
            cb_tmp_obj.pop_front(1);
            cb_max_obj.push_back(1);
        }

        // compute x - max(x)  — COL bcast: cb_max is 1 tile broadcast across Wt cols.
        // Reconfig: sub_bcast_cols_init_short_with_dt -> Input. pack_tile_with_dt -> Output.
        // Lifecycles: cb_in0 InputLifecycle::Bulk + Block; cb_max InputLifecycle::Bulk + Scalar (chain emits
        //   wait/pop(1) via window_1d<Scalar> — commit 14a5a61e462 made the
        //   OperandKind drive the wait count); cb_x_m_max OutputLifecycle::Bulk + Block.
        compute_kernel_lib::sub<
            cb_in0,
            cb_max,
            cb_x_m_max,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Block,    // A index
            compute_kernel_lib::InputLifecycle::Bulk,  // A lifecycle
            compute_kernel_lib::InputLifecycle::Bulk,  // B lifecycle
            compute_kernel_lib::OperandKind::Scalar,   // B index (cb_max broadcast)
            compute_kernel_lib::OutputLifecycle::Bulk>(Wt);

        // compute exp(x - max(x)) — split into 2 chains, same pattern as
        // moreh_softmax_h.cpp. cb_x_m_max held outside; cb_mask held outside;
        // cb_exps OutputLifecycle::Streaming per-tile.
        //
        // Reconfig: copy_tile_init_with_dt -> Input. pack_tile_with_dt -> Output.
        cb_x_m_max_obj.wait_front(Wt);
        compute_kernel_lib::eltwise_chain(
            Wt - 1,
            compute_kernel_lib::CopyTile<
                cb_x_m_max,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::CopyTileReconfig::Input>{},
#ifndef SOFTMAX
            compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_exps,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        compute_kernel_lib::eltwise_chain(
            1u,
            compute_kernel_lib::CopyTile<
                cb_x_m_max,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::TileOffset::Set>{Wt - 1},
#ifndef SOFTMAX
            compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::CopyTile<
                cb_mask,
                compute_kernel_lib::Dst::D1,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_exps,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

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

        // compute final result — COL bcast on cb_recipsumexps (1 tile).
        // LOG: out = (x-max) - log(sum_exp). cb_x_m_max chain-read (LOG owns it).
        // !LOG: out = exp(x-max) / sum_exp. cb_exps chain-read; cb_x_m_max held
        //   externally (chain doesn't touch it — uses InputLifecycle::CallerManaged-style outer
        //   wait/pop in !LOG path).
        // cb_x_m_max wait/pop wrap the chain symmetrically; chain uses
        // InputLifecycle::CallerManaged + Scalar in LOG path to avoid double-pop.
        // Reconfig: *_bcast_cols_init_short_with_dt + pack_tile_with_dt
        // -> Input + Output.
        cb_x_m_max_obj.wait_front(Wt);
#ifdef LOG
        compute_kernel_lib::eltwise_chain(
            Wt,
            compute_kernel_lib::BinaryFpu<
                cb_x_m_max,
                cb_recipsumexps,
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::BroadcastDim::Col,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                cb_out0,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::Output>{});
#else
        compute_kernel_lib::mul<
            cb_exps,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::OutputLifecycle::Bulk>(Wt);
#endif
        cb_x_m_max_obj.pop_front(Wt);
    }
}
