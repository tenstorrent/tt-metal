// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// silu(gate) * up in one SFPU pass over a gate / up tile pair in DST, drivable from the MATH or the PACK thread
// (minimal_matmul's fused SwiGLU runs it on the PACK thread, so it overlaps the math thread's next subblock). The
// pattern follows ckernel_sfpu_clamped_silu_glu.h and moe_gpt's swiglu_sfpu.h, without their clamps / alpha / up + 1.
//
// The sigmoid depends on DST and the output format:
// - fp32_dest_acc_en: silu_tile's accurate fp32 sigmoid (exp_accurate + 2 Newton steps).
// - bf16 DST, block-float output (block_float_output, set by the program descriptor for bfp8_b / bfp4_b): a
//   sigmoid sized for the output's 7-bit mantissas. exp(-gate) is Schraudolph's 2**(xlog2 - 127) with a linear mantissa
//   (_sfpu_exp_21f_bf16_ without its polynomial refinement; the bias shifted by 0.043 centres the error at ~3%), the
//   reciprocal is swiglu_recip below and nothing is rounded to bf16 in between (DST stores truncate).
//   About a third of silu_tile + mul_binary_tile's SFPU time, and no less accurate against an fp32 SwiGLU once the
//   output is bfp8.
// - bf16 DST, any other output: silu_tile + mul_binary_tile's bf16 sigmoid and roundings, since a bf16 output would
//   show the cheap sigmoid's error.

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_exp.h"  // _float_to_int32_for_exp_21f_
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

// The block-float sigmoid's reciprocal. Blackhole and Quasar have a one-instruction approximate reciprocal
// (SFPARECIP / SFPNONLINEAR, ~7-bit mantissa), which is all a block-float output needs. Wormhole has no such
// instruction, so it falls back to sfpu_reciprocal_iter's quadratic seed plus one Newton step - the reciprocal
// silu_tile's bf16 sigmoid already uses, and still far cheaper than that sigmoid's exp. Both are programmed by
// minimal_matmul_swiglu_init.
sfpi_inline sfpi::vFloat swiglu_recip(sfpi::vFloat denominator) {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)
    return sfpi::approx_recip(denominator);
#else
    return sfpu_reciprocal_iter<1>(denominator);
#endif
}

template <bool is_fp32_dest_acc_en, bool block_float_output, int ITERATIONS = 8>
inline void calculate_minimal_matmul_swiglu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = up * (gate * _sfpu_sigmoid_<true>(gate));
        } else if constexpr (block_float_output) {
            // xlog2 = -gate / ln2 + 127 (less the centring shift), clamped so the integer conversion cannot wrap: 0
            // gives exp = 0 (sigmoid 1), 255 gives +inf (sigmoid 0).
            sfpi::vFloat xlog2 = sfpi::clamp(gate * -1.4426950216293334961f + 126.9570f, 0.0f, 255.0f);
            sfpi::vFloat exp_neg_gate = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
            result = up * (gate * swiglu_recip(1.0f + exp_neg_gate));
        } else {
            sfpi::vFloat silu =
                sfpi::convert<sfpi::vFloat16b>(gate * _sfpu_sigmoid_<false>(gate), sfpi::RoundMode::Nearest);
            result = sfpi::convert<sfpi::vFloat16b>(silu * up, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// _sfpu_sigmoid_ takes its reciprocal from sfpu_reciprocal_iter, whose seed lives in vConstFloatPrgm0..2; nothing on
// the binary SFPU init path programs them. The block-float path's own constants are all SFPLOADI immediates, but on
// Wormhole its swiglu_recip is sfpu_reciprocal_iter too, so it needs the same seed.
inline void minimal_matmul_swiglu_init() { sigmoid_init</*APPROXIMATION_MODE=*/false>(); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void llk_minimal_matmul_swiglu_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::minimal_matmul_swiglu_init);
}

template <bool block_float_output>
inline void llk_minimal_matmul_swiglu(uint gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_minimal_matmul_swiglu,
        (DST_ACCUM_MODE, block_float_output, 8 /* ITERATIONS */),
        gate_tile,
        up_tile,
        out_tile,
        VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
