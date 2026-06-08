// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // sub
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Mask, Negative
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
            // mask cb_in0[0] (held -> HeldStream) with cb_mask (held externally -> CallerManaged).
            // copy_tile_init_with_dt -> Input; pack_tile_with_dt -> Output.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::
                    CopyTile<cb_in0, compute_kernel_lib::Dst::D0, compute_kernel_lib::InputLifecycle::HeldStream>{},
                compute_kernel_lib::
                    CopyTile<cb_mask, compute_kernel_lib::Dst::D1, compute_kernel_lib::InputLifecycle::CallerManaged>{},
                compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<cb_tmp>{});

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into cb_max via the helper.
            // cb_in0 holds all Wt tiles persistently for later steps, so use
            // WaitUpfrontNoPop — the helper waits for the slice it needs and never pops.
            compute_kernel_lib::
                reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                    cb_in0, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile (index Wt-1, no pop) and continue reducing
            // into cb_max via Accumulate. The accumulator and output are both cb_max:
            // the helper waits+pops the previous tile, then packs+pushes the new one.
            // mask cb_in0[Wt-1] (held -> HeldBulk + TileOffset::Set) with cb_mask (held -> CallerManaged).
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_in0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::HeldBulk,
                    compute_kernel_lib::CopyTileReconfig::Input,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::TileOffset::Set>{Wt - 1},
                compute_kernel_lib::
                    CopyTile<cb_mask, compute_kernel_lib::Dst::D1, compute_kernel_lib::InputLifecycle::CallerManaged>{},
                compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<cb_tmp>{});
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp,
                cb_max_scaler,
                cb_max,
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, /*iter=*/1));
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
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::OutputLifecycle::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>(Wt);

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
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::OperandKind::Block>{},
#ifndef SOFTMAX
            compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<cb_exps>{});

        compute_kernel_lib::eltwise_chain(
            1u,
            compute_kernel_lib::CopyTile<
                cb_x_m_max,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::TileOffset::Set>{Wt - 1},
#ifndef SOFTMAX
            compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::
                CopyTile<cb_mask, compute_kernel_lib::Dst::D1, compute_kernel_lib::InputLifecycle::CallerManaged>{},
            compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<cb_exps>{});

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
        compute_kernel_lib::sub<
            cb_x_m_max,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::OutputLifecycle::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>(Wt);
#else
        compute_kernel_lib::mul<
            cb_exps,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::OutputLifecycle::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>(Wt);
#endif
        cb_x_m_max_obj.pop_front(Wt);
    }
}
