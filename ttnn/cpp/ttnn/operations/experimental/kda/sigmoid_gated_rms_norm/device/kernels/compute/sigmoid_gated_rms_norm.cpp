// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

template <uint32_t Vt>
TT_KERNEL void compute(uint32_t wi_count) {
    compute_kernel_hw_startup(dfb::x, dfb::scaler, dfb::out);
    DataflowBuffer x(dfb::x);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer norm(dfb::norm);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer epsilon(dfb::epsilon);
    weight.wait_front(Vt);
    scaler.wait_front(1);
    epsilon.wait_front(1);
    for (uint32_t i = 0; i < wi_count; i++) {
        x.wait_front(Vt);
        gate.wait_front(Vt);

        // square(x) -> tmp
        ckl::square<
            ckl::input(dfb::x, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
            ckl::output(dfb::tmp, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(ckl::IterationShape::tiles(Vt));

        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        stats.wait_front(1);

        // rsqrt(stats + epsilon) -> inv
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::stats, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
                ckl::input(
                    dfb::epsilon,
                    ckl::BroadcastDim::None,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar)>{},
            ckl::Rsqrt<>{},
            ckl::PackTile<ckl::output(dfb::inv, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});

        inv.wait_front(1);

        // x * inv_rms -> norm
        ckl::mul<
            ckl::input(dfb::x, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
            ckl::input(
                dfb::inv,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Scalar),
            ckl::output(dfb::norm, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::tiles(Vt));

        norm.wait_front(Vt);
        x.pop_front(Vt);
        inv.pop_front(1);
        stats.pop_front(1);

        // norm * weight -> tmp
        ckl::mul<
            ckl::input(dfb::norm, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
            ckl::input(
                dfb::weight,
                ckl::BroadcastDim::Row,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block),
            ckl::output(dfb::tmp, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(ckl::IterationShape::tiles(Vt));

        tmp.wait_front(Vt);
        norm.pop_front(Vt);

        // sigmoid(gate) -> norm
        ckl::unary<
            ckl::Sigmoid<>,
            ckl::input(dfb::gate, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
            ckl::output(dfb::norm, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::tiles(Vt));

        norm.wait_front(Vt);
        gate.pop_front(Vt);

        // tmp * norm -> out
        ckl::mul<
            ckl::input(dfb::tmp, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
            ckl::input(dfb::norm, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
            ckl::output(dfb::out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(ckl::IterationShape::tiles(Vt));

        tmp.pop_front(Vt);
        norm.pop_front(Vt);
    }
}
