// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Half-width gate / up tiles (SE_HG): a gate/up weight tile holds 16 gate columns then the same 16 up columns, so the
// matmul output tile has gate in faces 0 / 2 and up in faces 1 / 3. calculate_glu_faces runs under VectorMode::C (the
// SFPU visits faces 0 and 2): it reads gate at the face, up one face on (+8 SFPU rows) of tile SRC and writes the
// activation into the left (HALF 0: faces 0 / 2) or right (HALF 1: faces 1 / 3) half of tile OUT. (Call it from one
// call site: two inlined copies in one loop trip the SFPI compiler's rvtt_synth_renumber pass.) Two units of one
// 32-column h tile: the first in place (left), the second into the first's right half (its up faces are consumed by
// then). Activations (SE_ACT): 0 SiLU-GLU, 2 SiTU-GLU (Kimi), 3 clamped SiLU-GLU (DeepSeek V4); the sigmoid is SiTU's
// (exp + Newton reciprocal, no vConstFloatPrgm; init: situ_glu_init).
#pragma once

#include <cstdint>

#include "ckernel_sfpu_situ_glu.h"
#include "ckernel_sfpu_clamped_silu_glu.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ACT, int ITERATIONS = 8>
inline void calculate_glu_faces(const uint src_tile, const uint /*unused*/, const uint out_tile, const uint half) {
    constexpr uint ts = 32;  // SFPU rows per tile (a face: 8)
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[src_tile * ts];
        sfpi::vFloat result;
        if constexpr (ACT == 2) {
            using C = SituGluConfigKimi;
            const sfpi::vFloat up = sfpi::dst_reg[src_tile * ts + 8];
            const sfpi::vFloat a =
                _sfpu_softcap_(gate, C::beta_gate, 1.0f / C::beta_gate) * _situ_glu_sigmoid_<is_fp32_dest_acc_en>(gate);
            result = a * _sfpu_softcap_(up, C::beta_up, 1.0f / C::beta_up);
        } else {
            static_assert(ACT == 0 || ACT == 3, "SE_HG: SiLU / SiTU / clamped SiLU GLU only");
            if constexpr (ACT == 3) {
                gate = sfpi::min(gate, ClampedSiluGluConfigDsV4::limit);
            }
            const sfpi::vFloat silu = gate * _situ_glu_sigmoid_<is_fp32_dest_acc_en>(gate);
            sfpi::vFloat up = sfpi::dst_reg[src_tile * ts + 8];
            if constexpr (ACT == 3) {
                up = sfpi::clamp(up, -ClampedSiluGluConfigDsV4::limit, ClampedSiluGluConfigDsV4::limit);
            }
            result = silu * up;
        }
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile * ts + half * 8] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
