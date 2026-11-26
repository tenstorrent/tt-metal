// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The face-wise GLU activation of half-width gate/up tiles (flat_options::half_width_gate_up, gate_up_compute.cpp).
//
// A half-width gate/up weight tile holds 16 gate columns, then the same 16 up columns, so the matmul's output tile has
// gate in faces 0 / 2 and up in faces 1 / 3:
//
//   output tile           h tile (out_tile)
//   +--------+--------+   +---------+---------+
//   | gate 0 | up 1   |   | half 0  | half 1  |
//   +--------+--------+   |  faces  |  faces  |
//   | gate 2 | up 3   |   |  0 / 2  |  1 / 3  |
//   +--------+--------+   +---------+---------+
//
// calculate_glu_faces runs under VectorMode::C (the SFPU visits faces 0 and 2): it reads gate at the face and up one
// face on (+8 SFPU rows) of tile src_tile, and writes act(gate, up) into the left (half 0: faces 0 / 2) or right
// (half 1: faces 1 / 3) half of tile out_tile. The two 16-column units of one 32-column h tile: the first in place
// (left half), the second into the first's right half (whose up faces are consumed by then).
//
// The activations are expert_activation::Kind's, with the model's constants (ExpertActivationConstants, from
// common/kernels/expert_activation.hpp); every one takes SiTU's sigmoid
// (exp and a Newton reciprocal, no programmable constants; init: situ_glu_init). Call it from one call site: two
// inlined copies in one loop trip the SFPI compiler's rvtt_synth_renumber pass.
#pragma once

#include <cstdint>

#include "ckernel_sfpu_clamped_silu_glu.h"
#include "ckernel_sfpu_situ_glu.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, expert_activation::Kind activation, int ITERATIONS = 8>
inline void calculate_glu_faces(const uint src_tile, const uint /*unused*/, const uint out_tile, const uint half) {
    using Kind = expert_activation::Kind;
    constexpr uint tile_rows = 32;  // SFPU rows per tile (a face: 8)
    constexpr uint up_face = 8;     // the up face's rows past the gate face's
    for (int row = 0; row < ITERATIONS; row++) {
        sfpi::vFloat gate = sfpi::dst_reg[src_tile * tile_rows];
        sfpi::vFloat result;
        if constexpr (activation == Kind::SwigluOai) {
            constexpr float alpha = ExpertActivationConstants::alpha, limit = ExpertActivationConstants::limit;
            gate = sfpi::min(gate, limit);
            const sfpi::vFloat up =
                sfpi::clamp(sfpi::vFloat(sfpi::dst_reg[src_tile * tile_rows + up_face]), -limit, limit);
            result = (up + 1.0f) * gate * _situ_glu_sigmoid_<is_fp32_dest_acc_en>(alpha * gate);
        } else if constexpr (activation == Kind::Situ) {
            using Config = ExpertActivationConstants;
            const sfpi::vFloat up = sfpi::dst_reg[src_tile * tile_rows + up_face];
            const sfpi::vFloat gate_part = _sfpu_softcap_(gate, Config::beta_gate, 1.0f / Config::beta_gate) *
                                           _situ_glu_sigmoid_<is_fp32_dest_acc_en>(gate);
            result = gate_part * _sfpu_softcap_(up, Config::beta_up, 1.0f / Config::beta_up);
        } else {
            static_assert(activation == Kind::Silu || activation == Kind::ClampedSilu, "an expert_activation::Kind");
            if constexpr (activation == Kind::ClampedSilu) {
                gate = sfpi::min(gate, ExpertActivationConstants::limit);
            }
            const sfpi::vFloat silu = gate * _situ_glu_sigmoid_<is_fp32_dest_acc_en>(gate);
            sfpi::vFloat up = sfpi::dst_reg[src_tile * tile_rows + up_face];
            if constexpr (activation == Kind::ClampedSilu) {
                up = sfpi::clamp(up, -ExpertActivationConstants::limit, ExpertActivationConstants::limit);
            }
            result = silu * up;
        }
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile * tile_rows + half * up_face] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
