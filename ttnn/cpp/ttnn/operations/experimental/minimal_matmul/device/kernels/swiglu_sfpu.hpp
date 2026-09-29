// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// silu(gate) * up in one SFPU pass over a gate / up tile pair in DST, drivable from the MATH or the PACK thread
// (minimal_matmul's fused SwiGLU runs it on the PACK thread, so it overlaps the math thread's next subblock). Same
// sigmoid and bf16 roundings as silu_tile followed by mul_binary_tile. The pattern follows
// ckernel_sfpu_clamped_silu_glu.h and moe_gpt's swiglu_sfpu.h, without their clamps / alpha / up + 1.

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

#if defined(SWIGLU_APPROX)
// swiglu_approx (bf16 dest only; fp32 dest keeps the exact path below): cheaper sigmoid, and no bf16 rounding of the
// intermediate silu (only the final result is rounded). e = Schraudolph exp(-gate): 2^(-gate/ln2) with a linear
// mantissa (exponent offset 0.0436 -> rel. error within +-3%), clamped; sigmoid = SFPARECIP(d) without a Newton step,
// with d = (1 + e) * (255/256) in one MAD, which re-centres SFPARECIP's downward bias (approx_recip(1.0) = 1 - 2^-7,
// so silu(x) for large x would otherwise come out 0.78% low).
template <int ITERATIONS = 8>
inline void calculate_minimal_matmul_swiglu_approx(
    const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
    sfpi::vFloat neg_one_ln2 = -EXP_21F_ONE_LN2, bias = 126.9564f;
    sfpi::vFloat rc = 0.99609375f;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat xlog2 = gate * neg_one_ln2 + bias;
        // [1, 254]: keeps the exponent field of the result normal and finite for any |gate| (exp_21f's [0, 255]
        // relies on its polynomial step to stay sane at the ends)
        xlog2 = sfpi::clamp(xlog2, 1.0f, 254.0f);
        sfpi::vFloat e = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
        sfpi::vFloat sig = sfpi::approx_recip(e * rc + rc);
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat result = (gate * sig) * up;
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}
#endif  // SWIGLU_APPROX

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_minimal_matmul_swiglu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
#if defined(SWIGLU_APPROX)
    if constexpr (!is_fp32_dest_acc_en) {
        calculate_minimal_matmul_swiglu_approx<ITERATIONS>(gate_tile_idx, up_tile_idx, out_tile_idx);
        return;
    }
#endif
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
    // Backport note: the exp's loop-invariant constants are hoisted into LRegs as calculate_silu does (this tree's
    // _sfpu_sigmoid_ overload; bit-identical arithmetic, 1/ln2 from vConstFloatPrgm1 = sigmoid_init<false>). up is
    // loaded after the sigmoid so the constants fit beside the live data.
    HoistedIf<!is_fp32_dest_acc_en> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    HoistedIf<is_fp32_dest_acc_en> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0;
    constexpr float p1 = EXP_FP32_P1;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat silu = gate * _sfpu_sigmoid_<is_fp32_dest_acc_en>(gate, c0, c1, c2, neg_ln2_hi, p0, p1);
        if constexpr (!is_fp32_dest_acc_en) {
            silu = sfpi::convert<sfpi::vFloat16b>(silu, sfpi::RoundMode::Nearest);
        }
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat result = silu * up;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// _sfpu_sigmoid_ takes its reciprocal from sfpu_reciprocal_iter, which needs vConstFloatPrgm0 = 2.0f; nothing on the
// binary SFPU init path programs it.
inline void minimal_matmul_swiglu_init() { sigmoid_init</*APPROXIMATION_MODE=*/false>(); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void llk_minimal_matmul_swiglu_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::minimal_matmul_swiglu_init);
}

inline void llk_minimal_matmul_swiglu(uint gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_minimal_matmul_swiglu,
        (DST_ACCUM_MODE, 8 /* ITERATIONS */),
        gate_tile,
        up_tile,
        out_tile,
        VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
