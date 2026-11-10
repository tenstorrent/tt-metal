// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// minimal_matmul's pack-thread SwiGLU (minimal_matmul/device/kernels/swiglu_sfpu.hpp) with a per-row scale: gate and
// up are both multiplied by the scale tile in DST first, silu(s * gate) * (s * up), written over the gate tile. The
// scale tile holds each row's scale in every column (an RMSNorm's 1/rms, applied after the matmul of the
// un-normalized rows). Same sigmoid choices as the original.

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ttnn/operations/experimental/minimal_matmul/device/kernels/swiglu_sfpu.hpp"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, bool block_float_output, int ITERATIONS = 8>
inline void calculate_swiglu_scaled(const uint gate_tile_idx, const uint up_tile_idx, const uint scale_tile_idx) {
    constexpr uint dst_tile_size = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat s = sfpi::dst_reg[scale_tile_idx * dst_tile_size];
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size] * s;
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size] * s;
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
        sfpi::dst_reg[gate_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

namespace ckernel {

template <bool block_float_output>
inline void lego_swiglu_scaled_tile(uint32_t gate_tile, uint32_t up_tile, uint32_t scale_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_swiglu_scaled,
        (DST_ACCUM_MODE, block_float_output, 8 /* ITERATIONS */),
        gate_tile,
        up_tile,
        scale_tile,
        VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH
