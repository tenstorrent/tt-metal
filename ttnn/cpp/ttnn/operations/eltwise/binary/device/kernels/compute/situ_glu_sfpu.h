// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//=============================================================================
// SiTU-GLU activation for SFPU (fused binary op).
//
//   situ_a  = beta_gate * tanh(gate / beta_gate) * sigmoid(gate)
//   up_half = beta_up   * tanh(up   / beta_up)
//   result  = situ_a * up_half
//
// gate and up are pinned in dst simultaneously, so the activation runs in one pass
// with no intermediate materialized to L1/DRAM.
//
// Usage:
//   PACK((llk_math_eltwise_binary_sfpu_situ_glu_init()));
//   PACK((llk_math_eltwise_binary_sfpu_situ_glu<false>(gate, up, out)));           // Kimi betas
//   PACK((llk_math_eltwise_binary_sfpu_situ_glu<false, MyConfig>(gate, up, out))); // custom
//
// Init: one tanh_init serves the whole op -- every half, sigmoid included, is a
// tanh polynomial.
//=============================================================================

#if defined(TRISC_PACK) || defined(TRISC_MATH)

// _sfpu_softcap_ has no Wormhole counterpart.
#if !defined(ARCH_BLACKHOLE)
#error "situ_glu_sfpu.h is implemented for Blackhole only"
#endif

#include "ckernel_sfpu_softcap.h"
#include "ckernel_sfpu_tanh.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

struct SituGluConfigKimi {
    static constexpr float beta_gate = 4.0f;
    static constexpr float beta_up = 25.0f;
};

// sigmoid(x) = (1 + tanh(x/2)) / 2, i.e. 0.5 * softcap(x, beta=0.5). Preferred over
// 1 / (1 + exp(-x)): it is branch-free where the reciprocal is predicated, and it reuses
// the tanh polynomial the two softcap halves already pay for.
//
// The polynomial clamps at |x/2| >= 3.375, so |x| >= 6.75 saturates to exactly 0 or 1
// instead of decaying. The 1.17e-3 that costs is below the 5.8e-2 the beta_up softcap
// already contributes, so it does not move the op's error.
sfpi_inline sfpi::vFloat _situ_glu_sigmoid_(sfpi::vFloat x) { return _sfpu_tanh_polynomial_(x * 0.5f) * 0.5f + 0.5f; }

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = SituGluConfigKimi>
inline void calculate_situ_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float beta_gate = Config::beta_gate;
    constexpr float inv_beta_gate = 1.0f / Config::beta_gate;
    constexpr float beta_up = Config::beta_up;
    constexpr float inv_beta_up = 1.0f / Config::beta_up;
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        // sigmoid takes the raw gate, not the capped value.
        sfpi::vFloat situ_a = _sfpu_softcap_(gate, beta_gate, inv_beta_gate) * _situ_glu_sigmoid_(gate);

        sfpi::vFloat result = situ_a * _sfpu_softcap_(up, beta_up, inv_beta_up);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// Always the polynomial coefficients, never the fp32-accurate tanh constants: _sfpu_softcap_
// and _situ_glu_sigmoid_ both read vConstFloatPrgm0-2 as polynomial coefficients regardless
// of dst mode.
inline void situ_glu_init() { tanh_init</*APPROXIMATION_MODE=*/false, /*is_fp32_dest_acc_en=*/false>(); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void llk_math_eltwise_binary_sfpu_situ_glu_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::situ_glu_init);
}

template <bool is_fp32_dest_acc_en = false, class Config = ckernel::sfpu::SituGluConfigKimi>
inline void llk_math_eltwise_binary_sfpu_situ_glu(
    uint gate_tile, uint32_t up_tile, uint32_t out_tile, VectorMode vector_mode = VectorMode::RC) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_situ_glu,
        (is_fp32_dest_acc_en, 8 /*ITERATIONS*/, Config),
        gate_tile,
        up_tile,
        out_tile,
        vector_mode);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
