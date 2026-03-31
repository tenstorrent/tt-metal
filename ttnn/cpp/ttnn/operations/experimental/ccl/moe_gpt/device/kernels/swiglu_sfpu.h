// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//=============================================================================
// GPT-OSS SwiGLU Activation for SFPU (runs on PACK thread)
//
// Formula:
//   gate_clamped = clamp(gate, max=clamp_limit)
//   up_clamped   = clamp(up, min=-clamp_limit, max=clamp_limit)
//   result       = (up_clamped + 1) * gate_clamped * sigmoid(alpha * gate_clamped)
//
// Usage:
//   // With default GPT-OSS config (alpha=1.702, clamp_limit=7.0):
//   PACK((llk_math_eltwise_binary_sfpu_swiglu_init<true>()));
//   PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false>(gate, up, out)));
//
//   // With custom config:
//   struct MyConfig { static constexpr float alpha = 1.0f; static constexpr float clamp_limit = 5.0f; };
//   PACK((llk_math_eltwise_binary_sfpu_swiglu_init<true>()));
//   PACK((llk_math_eltwise_binary_sfpu_swiglu<true, false, MyConfig>(gate, up, out)));
//
// This header is designed to be reusable across different models.
// Include it from a compute kernel that runs SFPU on the PACK or MATH thread.
//=============================================================================

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

namespace ckernel::sfpu {

//-----------------------------------------------------------------------------
// Configuration structs - add new models here
//-----------------------------------------------------------------------------
struct SwiGLUConfigGPTOSS {
    static constexpr float alpha = 1.702f;
    static constexpr float clamp_limit = 7.0f;
};

//-----------------------------------------------------------------------------
// Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
// Precision level matches dest accumulation mode.
//-----------------------------------------------------------------------------
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _swiglu_sigmoid_(sfpi::vFloat x) {
    sfpi::vFloat exp_neg_x;
    if constexpr (is_fp32_dest_acc_en) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }
    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x;
    if constexpr (is_fp32_dest_acc_en) {
        return _sfpu_reciprocal_<2>(denominator);
    } else {
        return _sfpu_reciprocal_<1>(denominator);
    }
}

//-----------------------------------------------------------------------------
// Core SwiGLU computation
//-----------------------------------------------------------------------------
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = SwiGLUConfigGPTOSS>
inline void calculate_swiglu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float alpha = Config::alpha;
    constexpr float clamp_limit = Config::clamp_limit;

    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        // Clamp gate: clamp(gate, max=clamp_limit)
        v_if(gate > clamp_limit) { gate = clamp_limit; }
        v_endif;

        // Clamp up: clamp(up, min=-clamp_limit, max=clamp_limit)
        v_if(up > clamp_limit) { up = clamp_limit; }
        v_endif;
        v_if(up < -clamp_limit) { up = -clamp_limit; }
        v_endif;

        // up = up + 1
        up = up + sfpi::vConst1;

        // sigmoid(alpha * gate)
        sfpi::vFloat alpha_gate = gate * alpha;
        sfpi::vFloat sig = _swiglu_sigmoid_<is_fp32_dest_acc_en>(alpha_gate);

        // result = (up + 1) * gate * sigmoid(alpha * gate)
        // Compute gate*sig first (bounded range, similar to SiLU), then multiply by up.
        sfpi::vFloat glu = gate * sig;
        sfpi::vFloat result = up * glu;

        // Round to bf16 if not in fp32 dest accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void swiglu_init() {
    // SwiGLU uses sigmoid internally, which needs reciprocal table init.
    _init_reciprocal_<false, false>();
}

}  // namespace ckernel::sfpu

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_swiglu_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(ckernel::sfpu::swiglu_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, class Config = ckernel::sfpu::SwiGLUConfigGPTOSS>
inline void llk_math_eltwise_binary_sfpu_swiglu(
    uint gate_tile, uint32_t up_tile, uint32_t out_tile, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_swiglu<is_fp32_dest_acc_en, 8, Config>, gate_tile, up_tile, out_tile, vector_mode);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
