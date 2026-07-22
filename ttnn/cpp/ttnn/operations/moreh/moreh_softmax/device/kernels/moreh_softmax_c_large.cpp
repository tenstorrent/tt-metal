// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"
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
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                ckl::copy<ckl::input(cb_in0), ckl::output(cb_max)>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::binary_sfpu<ckl::BinaryMax<>, ckl::input(cb_in0), ckl::input(cb_max), ckl::output(cb_max)>(
                    ckl::EltwiseShape::tiles(onetile));
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    ckl::input(cb_in0),
                    ckl::input(cb_max, ckl::InputLifecycle::HeldStream),
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_exps)>{});
#else
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    ckl::input(cb_in0),
                    ckl::input(cb_max, ckl::InputLifecycle::HeldStream),
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None>{},
                ckl::Negative<ckl::Dst::D0>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_exps)>{});
#endif

            if (i == 0) {
                ckl::copy<ckl::input(cb_exps), ckl::output(cb_add)>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<ckl::input(cb_add), ckl::input(cb_exps), ckl::output(cb_add)>(
                    ckl::EltwiseShape::tiles(onetile));
            }
        }

#ifdef LOG
        ckl::unary<ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>, ckl::input(cb_add), ckl::output(cb_recipsumexps)>(
            ckl::EltwiseShape::tiles(onetile));
#else
        ckl::unary<ckl::Recip<ckl::Dst::D0>, ckl::input(cb_add), ckl::output(cb_recipsumexps)>(
            ckl::EltwiseShape::tiles(onetile));
#endif

        cb_recipsumexps_obj.wait_front(onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            ckl::sub<
                ckl::input(cb_in0),
                ckl::input(cb_max, ckl::InputLifecycle::HeldStream),
                ckl::output(cb_tmp),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
            ckl::sub<
                ckl::input(cb_tmp),
                ckl::input(cb_recipsumexps, ckl::InputLifecycle::HeldStream),
                ckl::output(cb_out0),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
#else
            ckl::sub<
                ckl::input(cb_max, ckl::InputLifecycle::HeldStream),
                ckl::input(cb_in0),
                ckl::output(cb_tmp),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
            ckl::sub<
                ckl::input(cb_tmp),
                ckl::input(cb_recipsumexps, ckl::InputLifecycle::HeldStream),
                ckl::output(cb_out0),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
#endif
#else
#ifdef SOFTMAX
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    ckl::input(cb_in0),
                    ckl::input(cb_max, ckl::InputLifecycle::HeldStream),
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_exps)>{});
            ckl::mul<
                ckl::input(cb_exps),
                ckl::input(cb_recipsumexps, ckl::InputLifecycle::HeldStream),
                ckl::output(cb_out0),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
#else
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    ckl::input(cb_in0),
                    ckl::input(cb_max, ckl::InputLifecycle::HeldStream),
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::None>{},
                ckl::Negative<ckl::Dst::D0>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(cb_exps)>{});
            ckl::mul<
                ckl::input(cb_exps),
                ckl::input(cb_recipsumexps, ckl::InputLifecycle::HeldStream),
                ckl::output(cb_out0),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
