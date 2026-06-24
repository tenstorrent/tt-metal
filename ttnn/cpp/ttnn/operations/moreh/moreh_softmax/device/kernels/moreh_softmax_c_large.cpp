// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // BinaryMax
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

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
                ckl::copy<cb_in0, cb_max>(ckl::EltwiseShape::tiles(onetile));
            } else {
                // cb_max = max(cb_in0, cb_max) — same accumulator pattern as
                // moreh_norm_h/w ord_other (7e61967482a).
                ckl::binary_sfpu<ckl::BinaryMax<>, cb_in0, cb_max, cb_max>(ckl::EltwiseShape::tiles(onetile));
            }
        }

        // compute exp(x - max(x)) per C tile. No bcast since cb_max and cb_in0
        // are both full tiles. cb_max held outside loop (InputLifecycle::CallerManaged).
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    cb_in0,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldStream>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<cb_exps>{});
#else
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    cb_in0,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldStream>{},
                ckl::Negative<ckl::Dst::D0>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<cb_exps>{});
#endif

            // Accumulator over C-dim.
            if (i == 0) {
                ckl::copy<cb_exps, cb_add>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<cb_add, cb_exps, cb_add>(ckl::EltwiseShape::tiles(onetile));
            }
        }

        // log(sum) or 1/sum: single chain on cb_add -> cb_recipsumexps.
#ifdef LOG
        ckl::unary<ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>, cb_add, cb_recipsumexps>(
            ckl::EltwiseShape::tiles(onetile));
#else
        ckl::unary<ckl::Recip<ckl::Dst::D0>, cb_add, cb_recipsumexps>(ckl::EltwiseShape::tiles(onetile));
#endif

        // step 3, compute final result per C tile.
        cb_recipsumexps_obj.wait_front(onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum). Two chains.
            ckl::sub<
                cb_in0,
                cb_max,
                cb_tmp,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldStream>(ckl::EltwiseShape::tiles(onetile));
            ckl::sub<
                cb_tmp,
                cb_recipsumexps,
                cb_out0,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldStream>(ckl::EltwiseShape::tiles(onetile));
#else
            // -x + max - log(sum). Same as Sub(cb_max, cb_in0) followed by Sub.
            // cb_max held (pop0=0); cb_in0 popped (pop1=1).
            ckl::sub<cb_max, cb_in0, cb_tmp, ckl::BroadcastDim::None, ckl::InputLifecycle::HeldStream>(
                ckl::EltwiseShape::tiles(onetile));
            ckl::sub<
                cb_tmp,
                cb_recipsumexps,
                cb_out0,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldStream>(ckl::EltwiseShape::tiles(onetile));
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum. Sub+Exp folded; then Mul.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    cb_in0,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldStream>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<cb_exps>{});
            ckl::mul<
                cb_exps,
                cb_recipsumexps,
                cb_out0,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldStream>(ckl::EltwiseShape::tiles(onetile));
#else
            // rexp(x - max) / sum (softmin path).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    cb_in0,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldStream>{},
                ckl::Negative<ckl::Dst::D0>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<cb_exps>{});
            ckl::mul<
                cb_exps,
                cb_recipsumexps,
                cb_out0,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldStream>(ckl::EltwiseShape::tiles(onetile));
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
