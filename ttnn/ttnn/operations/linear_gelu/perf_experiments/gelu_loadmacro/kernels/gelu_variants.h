// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Pack-thread tanh-GELU variants for the gelu_loadmacro bake-off.
// Variant 0 is the op's current path (calculate_gelu_tanh_fast<true> via GeluTanhActivation<1>).
#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_gelu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "gelu_lm_sfpu.h"
#endif

template <uint32_t V>
ALWI void gv_init() {
    if constexpr (V == 0) {
        PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(ckernel::sfpu::gelu_tanh_fast_init));
    } else {
        PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(ckernel::sfpu::gelu_lm_init<V>));
    }
}

template <uint32_t V>
ALWI void gv_apply(uint32_t idst) {
    if constexpr (V == 0) {
        PACK(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gelu_tanh_fast, (DST_ACCUM_MODE), idst, VectorMode::RC));
    } else {
        PACK(_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::gelu_lm_face<V>, idst, VectorMode::RC));
    }
}
