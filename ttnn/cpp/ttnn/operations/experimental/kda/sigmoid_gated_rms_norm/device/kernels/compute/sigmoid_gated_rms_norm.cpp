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

// Inputs are waited for and popped by the caller; each helper reserves and publishes its whole output.
constexpr ckl::InputSpec resident(uint32_t dfb_id) {
    return ckl::input(dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
}

constexpr ckl::OutputSpec published(uint32_t dfb_id) {
    return ckl::output(dfb_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);
}

template <typename DFB>
void square(uint32_t n, DFB& tmp) {
    DataflowBuffer x(dfb::x);
    ckl::square<resident(dfb::x), published(dfb::tmp)>(ckl::IterationShape::tiles(n), x, tmp);
}

template <typename DFB>
void inverse_rms(DFB& inv) {
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer epsilon(dfb::epsilon);
    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Add, resident(dfb::stats), resident(dfb::epsilon)>{stats, epsilon},
        ckl::Rsqrt<>{},
        ckl::PackTile<published(dfb::inv)>{inv});
}

template <typename DFB>
void scale_by_inverse_rms(uint32_t Vt, DFB& norm) {
    DataflowBuffer x(dfb::x);
    DataflowBuffer inv(dfb::inv);
    ckl::mul<
        resident(dfb::x),
        ckl::input(dfb::inv, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None),
        published(dfb::norm)>(ckl::IterationShape::tiles(Vt), x, inv, norm);
}

template <typename DFB>
void apply_weight(uint32_t Vt, DFB& tmp) {
    DataflowBuffer norm(dfb::norm);
    DataflowBuffer weight(dfb::weight);
    ckl::mul<
        resident(dfb::norm),
        ckl::input(
            dfb::weight,
            ckl::BroadcastDim::Row,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block),
        published(dfb::tmp)>(ckl::IterationShape::tiles(Vt), norm, weight, tmp);
}

// Flattened so the unpacker init folds to this buffer's tile shape. Inlined into the full work-item loop, it would
// exceed GCC's function-growth limit and stay a call that decodes the shape at run time, at a stage boundary on the
// critical path.
template <typename DFB>
__attribute__((flatten)) void activate_gate(uint32_t Vt, DFB& norm) {
    DataflowBuffer gate(dfb::gate);
    ckl::unary<ckl::Sigmoid<>, resident(dfb::gate), published(dfb::norm)>(ckl::IterationShape::tiles(Vt), gate, norm);
}

template <typename DFB>
void multiply_output(uint32_t Vt, DFB& out) {
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer norm(dfb::norm);
    ckl::mul<resident(dfb::tmp), resident(dfb::norm), published(dfb::out)>(
        ckl::IterationShape::tiles(Vt), tmp, norm, out);
}

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
    DataflowBuffer out(dfb::out);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer epsilon(dfb::epsilon);
    weight.wait_front(Vt);
    scaler.wait_front(1);
    epsilon.wait_front(1);
    for (uint32_t i = 0; i < wi_count; i++) {
        x.wait_front(Vt);
        gate.wait_front(Vt);
        square(Vt, tmp);
        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        stats.wait_front(1);
        inverse_rms(inv);
        inv.wait_front(1);
        scale_by_inverse_rms(Vt, norm);
        norm.wait_front(Vt);
        x.pop_front(Vt);
        inv.pop_front(1);
        stats.pop_front(1);
        apply_weight(Vt, tmp);
        tmp.wait_front(Vt);
        norm.pop_front(Vt);
        activate_gate(Vt, norm);
        norm.wait_front(Vt);
        gate.pop_front(Vt);
        multiply_output(Vt, out);
        tmp.pop_front(Vt);
        norm.pop_front(Vt);
    }
}
