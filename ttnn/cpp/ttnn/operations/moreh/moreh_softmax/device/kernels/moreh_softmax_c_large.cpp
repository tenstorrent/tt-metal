// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // BinaryMax
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = tt::CBIndex::c_26;
    constexpr auto cb_max = tt::CBIndex::c_27;
    CircularBuffer cb_max_obj(cb_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_in0, cb_exps, cb_out0);

    for (uint32_t n = 0; n < N; ++n) {
        // find max via running BinaryMax across C-dim.
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                // Seed cb_max with first cb_in0 tile.
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_in0,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::PackTile<
                        cb_max,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            } else {
                // cb_max = max(cb_in0, cb_max) — same accumulator pattern as
                // moreh_norm_h/w ord_other (7e61967482a).
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_in0,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::CopyTile<
                        cb_max,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::BinaryMax<
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_max,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }
        }

        // compute exp(x - max(x)) per C tile. No bcast since cb_max and cb_in0
        // are both full tiles. cb_max held outside loop (InputLifecycle::CallerManaged).
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::Exp<
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_exps,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#else
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
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
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#endif

            // Accumulator over C-dim.
            if (i == 0) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_exps,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::PackTile<
                        cb_add,
                        compute_kernel_lib::OutputLifecycle::Streaming,
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
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        cb_add,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }
        }

        // log(sum) or 1/sum: single chain on cb_add -> cb_recipsumexps.
#ifdef LOG
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_add,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::Log<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_recipsumexps,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
#else
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_add,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_recipsumexps,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
#endif

        // step 3, compute final result per C tile.
        cb_recipsumexps_obj.wait_front(onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum). Two chains.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_tmp,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_tmp,
                    cb_recipsumexps,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#else
            // -x + max - log(sum). Same as Sub(cb_max, cb_in0) followed by Sub.
            // cb_max held (pop0=0); cb_in0 popped (pop1=1).
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_max,
                    cb_in0,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_tmp,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_tmp,
                    cb_recipsumexps,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum. Sub+Exp folded; then Mul.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::Exp<
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_exps,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_exps,
                    cb_recipsumexps,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#else
            // rexp(x - max) / sum (softmin path).
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_max,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
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
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_exps,
                    cb_recipsumexps,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
