// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Expert activation on the packer's SFPU over DEST: dst 0 = act(W0 x) * (W1 x) for the first (W0, W1) pair, dst 2
// for the second (dst 1 and 3 hold W1 x). Shared by the ring kernel (compute.cpp) and the expert rows kernel.
#pragma once

#include "../hostdevcommon/config.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// Need these headers for running SFPU on PACK thread
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_gelu.h"
#endif

namespace moe_activation {

// Note GELU gets init'd at each iteration in its PackActivation specialization
template <ttnn::experimental::prim::detail::MoEActivationFunction activation>
inline void pack_init_activation() {};

template <>
inline void pack_init_activation<ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU>() {
    PACK((llk_math_eltwise_binary_sfpu_swiglu_init()));
};

template <>
inline void pack_init_activation<ttnn::experimental::prim::detail::MoEActivationFunction::CLAMPED_SILU>() {
    PACK((llk_math_eltwise_binary_sfpu_swiglu_init()));
};

template <>
inline void pack_init_activation<ttnn::experimental::prim::detail::MoEActivationFunction::SILU>() {
    PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (true /*APPROXIMATE*/)));
};

// kPairs = 1: only the first pair holds data -- the half block-column of a ring core with an odd gate/up column count.
template <ttnn::experimental::prim::detail::MoEActivationFunction activation, uint32_t kPairs>
struct PackActivation {
    static inline void compute() {}
};

template <uint32_t kPairs>
struct PackActivation<ttnn::experimental::prim::detail::MoEActivationFunction::SILU, kPairs> {
    static inline void compute() {
        PACK(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_silu,
            (DST_ACCUM_MODE, 8 /*ITERATIONS*/),
            0 /*DST_IDX*/,
            ::ckernel::VectorMode::RC));
        if constexpr (kPairs == 2) {
            PACK(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_silu,
                (DST_ACCUM_MODE, 8 /*ITERATIONS*/),
                2 /*DST_IDX*/,
                ::ckernel::VectorMode::RC));
        }

        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/, DST_ACCUM_MODE),
            0 /*DST_IN0*/,
            1 /*DST_IN1*/,
            0 /*DST_OUT*/,
            ::ckernel::VectorMode::RC)));
        if constexpr (kPairs == 2) {
            PACK((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sfpu_binary,
                (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/, DST_ACCUM_MODE),
                2 /*DST_IN0*/,
                3 /*DST_IN1*/,
                2 /*DST_OUT*/,
                ::ckernel::VectorMode::RC)));
        }
    }
};

template <uint32_t kPairs>
struct PackActivation<ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU, kPairs> {
    static inline void compute() {
        PACK((llk_math_eltwise_binary_sfpu_swiglu<DST_ACCUM_MODE>(0, 1, 0)));
        if constexpr (kPairs == 2) {
            PACK((llk_math_eltwise_binary_sfpu_swiglu<DST_ACCUM_MODE>(2, 3, 2)));
        }
    }
};

#ifdef TRISC_PACK
// silu(min(gate, limit)) * clamp(up, -limit, limit)
struct ClampedSiluConfig {
    static constexpr float alpha = 1.0f;
    static constexpr float clamp_limit =
        __builtin_bit_cast(float, get_named_compile_time_arg_val("activation_limit_bits"));
};
#endif

template <uint32_t kPairs>
struct PackActivation<ttnn::experimental::prim::detail::MoEActivationFunction::CLAMPED_SILU, kPairs> {
    static inline void compute() {
        PACK((llk_math_eltwise_binary_sfpu_swiglu<DST_ACCUM_MODE, ClampedSiluConfig, /*AddUpBias=*/false>(0, 1, 0)));
        if constexpr (kPairs == 2) {
            PACK(
                (llk_math_eltwise_binary_sfpu_swiglu<DST_ACCUM_MODE, ClampedSiluConfig, /*AddUpBias=*/false>(2, 3, 2)));
        }
    }
};

template <uint32_t kPairs>
struct PackActivation<ttnn::experimental::prim::detail::MoEActivationFunction::GELU, kPairs> {
    static inline void compute() {
        // GELU programs an SFPU LUT (gelu_init). The trailing binary MUL below clobbers that LUT,
        // so when the activation loop runs >1 iteration per chunk (tiles_per_step > 2, which happens
        // for ring sizes where ceil(Nt/ring) is odd — e.g. gemma at ring=8) the next iteration's
        // gelu reads a stale LUT and produces garbage. Re-init the LUT here so every gelu is valid.
        // SILU/SWIGLU don't use this LUT, so they keep their cheaper once-per-chunk init.
        PACK((llk_math_eltwise_unary_sfpu_init<SfpuType::gelu>(ckernel::sfpu::gelu_init<true, DST_ACCUM_MODE>)));
        PACK(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_gelu,
            (true /*APPROXIMATE*/, DST_ACCUM_MODE, 8 /*ITERATIONS*/),
            0 /*DST_IDX*/,
            ::ckernel::VectorMode::RC));
        if constexpr (kPairs == 2) {
            PACK(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_gelu,
                (true /*APPROXIMATE*/, DST_ACCUM_MODE, 8 /*ITERATIONS*/),
                2 /*DST_IDX*/,
                ::ckernel::VectorMode::RC));
        }

        PACK((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/, DST_ACCUM_MODE),
            0 /*DST_IN0*/,
            1 /*DST_IN1*/,
            0 /*DST_OUT*/,
            ::ckernel::VectorMode::RC)));
        if constexpr (kPairs == 2) {
            PACK((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sfpu_binary,
                (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/, DST_ACCUM_MODE),
                2 /*DST_IN0*/,
                3 /*DST_IN1*/,
                2 /*DST_OUT*/,
                ::ckernel::VectorMode::RC)));
        }
    }
};

}  // namespace moe_activation
