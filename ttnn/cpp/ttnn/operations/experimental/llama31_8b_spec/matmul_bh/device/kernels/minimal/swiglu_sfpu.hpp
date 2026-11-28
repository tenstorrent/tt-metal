// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat swiglu_recip(sfpi::vFloat denominator) {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)
    return sfpi::approx_recip(denominator);
#else
    return sfpu_reciprocal_iter<1>(denominator);
#endif
}

template <bool is_fp32_dest_acc_en, bool block_float_output, int ITERATIONS = 8>
inline void calculate_minimal_matmul_swiglu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            result = up * (gate * _sfpu_sigmoid_<true>(gate));
        } else if constexpr (block_float_output) {
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

inline void minimal_matmul_swiglu_init() { sigmoid_init<false>(); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void minimal_matmul_swiglu_tile_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::minimal_matmul_swiglu_init);
}

template <bool block_float_output>
inline void minimal_matmul_swiglu_tile(uint32_t gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_minimal_matmul_swiglu,
        (DST_ACCUM_MODE, block_float_output, 8),
        gate_tile,
        up_tile,
        out_tile,
        VectorMode::RC);
}

}  // namespace ckernel

#endif
