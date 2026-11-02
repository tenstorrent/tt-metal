// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// Routes GeluTanhActivation<GV_P0> (an otherwise unused param value) to the gelu_loadmacro variant GV_V, so the
// op's compute kernel and the shared matmul_block helper run unchanged apart from the SFPU body.
#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "GV_KERNELS_DIR/gelu_lm_sfpu.h"
#endif

namespace compute_kernel_lib {

template <>
struct ActivationInitHelper<KernelActivation::GELU_TANH, GV_P0, 0> {
    FORCE_INLINE static void init() {
        PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(ckernel::sfpu::gelu_lm_init<GV_V>));
    }
};

template <>
struct ActivationApplyHelper<KernelActivation::GELU_TANH, GV_P0, 0, 0> {
    FORCE_INLINE static void apply(uint32_t tile_index) {
        PACK(_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::gelu_lm_face<GV_V>, tile_index, VectorMode::RC));
    }
};

}  // namespace compute_kernel_lib
