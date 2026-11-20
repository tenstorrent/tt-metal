// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"

// The output gate's factors g = sigmoid(gate) * weight of the gated output norm, formed where the gate rows are: on a
// head's cores, and on a head core's helper for the head core.
namespace kda_gate {

// The norm weight's row 0 broadcast down every row, for the elementwise product with each output row.
template <uint32_t cb_weight, uint32_t cb_weight_rows, uint32_t kt>
void broadcast_weight() {
    namespace ckl = compute_kernel_lib;
    DataflowBuffer weight(cb_weight);
    DataflowBuffer weight_rows(cb_weight_rows);
    ckl::unary_bcast<
        ckl::BroadcastDim::Row,
        ckl::input(cb_weight, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::output(cb_weight_rows, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(kt), weight, weight_rows);
    cb_wait_front(cb_weight_rows, kt);
}

// Each of the next `chunks` gate rows into its chunk's g = sigmoid(gate) * weight.
template <uint32_t cb_gate, uint32_t cb_weight_rows, uint32_t cb_gate_factor, uint32_t kt, bool fp32_output>
void gate_factors(uint32_t chunks) {
    namespace ckl = compute_kernel_lib;
    DataflowBuffer gate(cb_gate);
    DataflowBuffer weight_rows(cb_weight_rows);
    DataflowBuffer gate_factor(cb_gate_factor);
    constexpr auto gate_row =
        ckl::input(cb_gate, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block);
    constexpr auto weight_row =
        ckl::input(cb_weight_rows, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
    constexpr auto gate_factor_row = ckl::output(cb_gate_factor, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);
    for (uint32_t c = 0; c < chunks; ++c) {
        if constexpr (fp32_output) {
            // The product on the SFPU, in FP32: the FPU's DEST reuse would round sigmoid(gate) to srcA's format.
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(kt).block_size(kt),
                ckl::CopyTile<gate_row, ckl::Dst::D0>{gate},
                ckl::Sigmoid<ckl::Dst::D0>{},
                ckl::CopyTile<weight_row, ckl::Dst::D1>{weight_rows},
                ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                ckl::PackTile<gate_factor_row, ckl::Dst::D0>{gate_factor});
        } else {
            // A BF16 g needs only the BF16-accurate sigmoid, and the FPU product reusing DEST.
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(kt).block_size(kt),
                ckl::CopyTile<gate_row>{gate},
                ckl::Sigmoid<ckl::Dst::D0, ckl::Approx::Exact, ckl::SigmoidPrecision::Bf16>{},
                ckl::DestReuseBinary<ckl::BinaryFpuOp::Mul, weight_row, ckl::DestReuseType::DEST_TO_SRCA>{weight_rows},
                ckl::PackTile<gate_factor_row>{gate_factor});
        }
    }
}

}  // namespace kda_gate
