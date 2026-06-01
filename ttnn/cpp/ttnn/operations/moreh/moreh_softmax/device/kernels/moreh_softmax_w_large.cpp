// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    CircularBuffer cb_in0_obj(cb_in0);
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = tt::CBIndex::c_26;
    constexpr auto cb_max = tt::CBIndex::c_27;
    CircularBuffer cb_max_obj(cb_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;
    CircularBuffer cb_tmp_obj(cb_tmp);

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
        // find max
        if (Wt == 1) {
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/1, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                cb_tmp, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: pop-as-we-go reduce of Wt-1 tiles into cb_max.
            // Not via reduce helper because helper's WaitUpfrontNoPop would buffer
            // all Wt-1 tiles; this pattern pops one-at-a-time to keep CB pressure
            // low under the per-tile streaming reader.
            cb_max_obj.reserve_back(onetile);

            tile_regs_acquire();
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_in0, cb_max_scaler);
#endif
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, cb_max);
            for (uint32_t w = 0; w < Wt - 1; ++w) {
                cb_in0_obj.wait_front(onetile);
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in0, cb_max_scaler, 0, 0, dst0);
                cb_in0_obj.pop_front(onetile);
            }
            reduce_uninit();
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_max);
            tile_regs_release();

            cb_max_obj.push_back(onetile);

            // Phase 2: merge the masked last tile into cb_max.
            mask_tile_to_cb(cb_in0, cb_mask, cb_tmp, 0, 0, /*pop0=*/1, /*popm=*/0);

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

        // step 1: exp(x - max) accumulator over w. cb_in0 streaming + Scalar (pop=1).
        // cb_max CallerManaged (held outside loop). cb_mask CallerManaged (popm=0).
        for (uint32_t w = 0; w < Wt; ++w) {
            // compute exp(x - max) [+ mask if last] OR rexp[+ mask] (!SOFTMAX last).
            if (w == Wt - 1) {
#ifdef SOFTMAX
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_in0,
                        cb_max,
                        compute_kernel_lib::BinaryFpuOp::Sub,
                        compute_kernel_lib::BroadcastDim::Col,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::HeldStream,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::Exp<
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::CopyTile<
                        cb_mask,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::HeldStream,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_exps,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#else
                // rexp + mask (no sub); matches rexp_tile_and_mask_tile_to_cb.
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_in0,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::Exp<
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::CopyTile<
                        cb_mask,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::HeldStream,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_exps,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#endif
            } else {
#ifdef SOFTMAX
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_in0,
                        cb_max,
                        compute_kernel_lib::BinaryFpuOp::Sub,
                        compute_kernel_lib::BroadcastDim::Col,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::HeldStream,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::Exp<
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_exps,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#else
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_in0,
                        cb_max,
                        compute_kernel_lib::BinaryFpuOp::Sub,
                        compute_kernel_lib::BroadcastDim::Col,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::HeldStream,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::Exp<
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Approx::Exact,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_exps,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#endif
            }

            // Accumulator over w. Seed copy on first iteration.
            if (w == 0) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_exps,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::PackTile<
                        cb_add,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            } else {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_add,
                        cb_exps,
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        cb_add,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }
        }

#ifdef LOG
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_add,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    log_tile_init();
                    log_tile(dst_idx);
                });
#else
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_add,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });
#endif

        // step 3, compute final result per tile.
        for (uint32_t w = 0; w < Wt; w += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum). Two chains (cb_in0 -> cb_tmp -> cb_out0).
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::Col,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_tmp,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_tmp,
                    cb_recipsumexps,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::Col,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#else
            // logsoftmin not implemented in original.
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) * 1/sum. Sub+Exp folded; then Mul by recip.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::Col,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::Exp<
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_exps,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_exps,
                    cb_recipsumexps,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::Col,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#else
            // rexp(x - max) / sum (softmin path).
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::Col,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::Exp<
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_exps,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_exps,
                    cb_recipsumexps,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::Col,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
