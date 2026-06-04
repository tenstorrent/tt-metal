// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
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

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    cb_mask_obj.wait_front(onetile);
    cb_max_scaler_obj.wait_front(onetile);
    cb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Ht == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL>(
                cb_tmp, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            compute_kernel_lib::
                reduce<PoolType::MAX, ReduceDim::REDUCE_COL, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                    cb_in0, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, Ht - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL>(
                cb_tmp,
                cb_max_scaler,
                cb_max,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, 1));  // iteration=1, reload from cb_max
        }

        // compute x - max(x)  — ROW bcast: cb_max is 1 tile broadcast across Ht rows.
        // Reconfig audit: sub_bcast_rows_init_short_with_dt reconfigs srca/srcb -> Input.
        //   pack_tile_with_dt -> Output.
        // Lifecycles: cb_in0 InputLifecycle::Bulk + Block (chain owns wait Ht + pop Ht). cb_max
        //   InputLifecycle::Bulk + Scalar — chain emits cb_wait_front(cb_max, 1) thanks to the
        //   OperandKind-aware window_1d helper. cb_x_m_max OutputLifecycle::Bulk + Block.
        compute_kernel_lib::sub<
            cb_in0,
            cb_max,
            cb_x_m_max,
            compute_kernel_lib::BroadcastDim::Row,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::OutputLifecycle::Bulk>(Ht);

        // compute exp(x - max(x)). Original per-tile copy + (Negative if !SOFTMAX)
        // + Exp + (Mask on last tile) + pack with cb_exps reserve(Ht) upfront and
        // push(Ht) at end. Split into 2 chains:
        //   A (Ht-1 iters, non-last): CopyTile(cb_x_m_max @ i) + ... + Exp + Pack
        //   B (1 iter, last with mask): CopyTile(cb_x_m_max @ Ht-1) + ... + Exp +
        //                                CopyTile(cb_mask) + Mask + Pack
        //
        // cb_x_m_max InputLifecycle::CallerManaged + Block (held outside; sequential read 0..Ht-1
        // across iters; compute_kernel_lib::TileOffset::Set(Ht-1) for the last tile).
        // cb_mask InputLifecycle::CallerManaged + Scalar (held outside via line 42 wait_front(1)).
        // cb_exps OutputLifecycle::Streaming (per-tile reserve+push replaces upfront reserve+
        // push; net Ht tiles pushed matches original).
        //
        // Reconfig: copy_tile_init_with_dt -> CopyTileReconfig::Input.
        // pack_tile_with_dt -> PackTileReconfig::Output.
        cb_x_m_max_obj.wait_front(Ht);
        compute_kernel_lib::eltwise_chain(
            Ht - 1,
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
                compute_kernel_lib::TileOffset::Set>{Ht - 1},
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
            reduce<PoolType::SUM, ReduceDim::REDUCE_COL, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_exps,
                cb_sum_scaler,
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
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::col(Ht),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });
#endif

        // compute final result — ROW bcast on cb_recipsumexps (1 tile).
        // LOG path: out = (x - max) - log(sum_exp). Reads cb_x_m_max (held by
        //   external wait/pop) and cb_recipsumexps.
        // !LOG path: out = exp(x-max) / sum_exp. Reads cb_exps (chain-owned bulk
        //   wait+pop) and cb_recipsumexps; cb_x_m_max held externally because
        //   the chain doesn't touch it.
        // cb_x_m_max wait/pop wrap the chain symmetrically in both paths (chain
        // uses InputLifecycle::CallerManaged on it in LOG path).
        // Reconfig: *_bcast_rows_init_short_with_dt -> Input.
        //   pack_tile_with_dt -> Output.
        cb_x_m_max_obj.wait_front(Ht);
#ifdef LOG
        compute_kernel_lib::sub<
            cb_x_m_max,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Row,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::OutputLifecycle::Bulk>(Ht);
#else
        compute_kernel_lib::mul<
            cb_exps,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Row,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::InputLifecycle::Bulk,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::OutputLifecycle::Bulk>(Ht);
#endif
        cb_x_m_max_obj.pop_front(Ht);
    }
}
