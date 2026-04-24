// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "internal/risc_attribs.h"
#include <cstring>  // for memcpy

using ttnn::operations::matmul::KernelActivation;

// Helper templates to select activation variants based on parameters
template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0>
struct ActivationInitHelper {
    // Compile-time validation
    static_assert(
        ACT == KernelActivation::NONE || ACT == KernelActivation::SILU || ACT == KernelActivation::TANH ||
            ACT == KernelActivation::GELU || ACT == KernelActivation::RELU6 || ACT == KernelActivation::SIGMOID ||
            ACT == KernelActivation::HARDSIGMOID || ACT == KernelActivation::HARDTANH ||
            ACT == KernelActivation::SELU || ACT == KernelActivation::SOFTPLUS,
        "Unsupported KernelActivation type for fused activation init");

    FORCE_INLINE static void init() {
        if constexpr (ACT == KernelActivation::SILU) {
            silu_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::TANH) {
            // PARAM0: 0 = accurate, non-zero = fast
            if constexpr (PARAM0 != 0) {
                tanh_tile_init_pack<true>();
            } else {
                tanh_tile_init_pack<false>();
            }
        } else if constexpr (ACT == KernelActivation::GELU) {
            // PARAM0: 0 = accurate, non-zero = fast
            if constexpr (PARAM0 != 0) {
                gelu_tile_init_pack<true>();
            } else {
                gelu_tile_init_pack<false>();
            }
        } else if constexpr (ACT == KernelActivation::RELU6) {
            relu_max_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::SIGMOID) {
            // Enhanced: PARAM1 is fast_approximate flag
            if constexpr (PARAM1 != 0) {
                sigmoid_tile_init_pack<true>();
            } else {
                sigmoid_tile_init_pack<false>();
            }
        } else if constexpr (ACT == KernelActivation::HARDSIGMOID) {
            hardsigmoid_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::HARDTANH) {
            hardtanh_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::SELU) {
            selu_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::SOFTPLUS) {
            softplus_tile_init_pack();
        }
    }
};

template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0>
struct ActivationApplyHelper {
    // Compile-time validation
    static_assert(
        ACT == KernelActivation::NONE || ACT == KernelActivation::SILU || ACT == KernelActivation::TANH ||
            ACT == KernelActivation::GELU || ACT == KernelActivation::RELU6 || ACT == KernelActivation::SIGMOID ||
            ACT == KernelActivation::HARDSIGMOID || ACT == KernelActivation::HARDTANH ||
            ACT == KernelActivation::SELU || ACT == KernelActivation::SOFTPLUS,
        "Unsupported KernelActivation type for fused activation apply");

    // Parameter-specific validation
    static_assert(
        ACT != KernelActivation::SIGMOID || PARAM0 <= 2,
        "SIGMOID PARAM0 must be 0 (RC), 1 (R), or 2 (C) for vector mode");

    static_assert(
        ACT != KernelActivation::SOFTPLUS || PARAM0 != 0,
        "SOFTPLUS PARAM0 (beta) must be non-zero to avoid division by zero");

    FORCE_INLINE static void apply(uint32_t tile_index) {
        if constexpr (ACT == KernelActivation::SILU) {
            silu_tile_pack(tile_index);
        } else if constexpr (ACT == KernelActivation::TANH) {
            // PARAM0: 0 = accurate, non-zero = fast
            if constexpr (PARAM0 != 0) {
                tanh_tile_pack<true>(tile_index);
            } else {
                tanh_tile_pack<false>(tile_index);
            }
        } else if constexpr (ACT == KernelActivation::GELU) {
            // PARAM0: 0 = accurate, non-zero = fast
            if constexpr (PARAM0 != 0) {
                gelu_tile_pack<true>(tile_index);
            } else {
                gelu_tile_pack<false>(tile_index);
            }
        } else if constexpr (ACT == KernelActivation::RELU6) {
            // PARAM0 is the max value (as uint32_t bit pattern)
            // Default to 6.0 if PARAM0 is 0
            constexpr uint32_t max = (PARAM0 != 0) ? PARAM0 : 0x40c00000u;
            relu_max_tile_pack(tile_index, max);
        } else if constexpr (ACT == KernelActivation::SIGMOID) {
            // Enhanced: PARAM0 is vector mode, PARAM1 is fast_approximate
            constexpr int vec_mode = (PARAM0 == 1) ? VectorMode::R : (PARAM0 == 2) ? VectorMode::C : VectorMode::RC;
            if constexpr (PARAM1 != 0) {
                sigmoid_tile_pack<vec_mode, true>(tile_index);
            } else {
                sigmoid_tile_pack<vec_mode, false>(tile_index);
            }
        } else if constexpr (ACT == KernelActivation::HARDSIGMOID) {
            hardsigmoid_tile_pack(tile_index);
        } else if constexpr (ACT == KernelActivation::HARDTANH) {
            hardtanh_tile_pack(tile_index, PARAM0, PARAM1);
        } else if constexpr (ACT == KernelActivation::SELU) {
            // PARAM0 is alpha, PARAM1 is lambda
            selu_tile_pack(tile_index, PARAM0, PARAM1);
        } else if constexpr (ACT == KernelActivation::SOFTPLUS) {
            // PARAM0 is beta, PARAM1 is threshold
            uint32_t beta = PARAM0;
            float beta_f;
            memcpy(&beta_f, &beta, sizeof(float));
            float beta_recip_f = 1.0f / beta_f;
            uint32_t beta_reciprocal;
            memcpy(&beta_reciprocal, &beta_recip_f, sizeof(uint32_t));
            softplus_tile_pack(tile_index, PARAM0, beta_reciprocal, PARAM1);
        }
    }
};

template <KernelActivation ACT>
FORCE_INLINE void init_sfpu_activation_pack() {
    ActivationInitHelper<ACT, 0, 0>::init();
}

template <KernelActivation ACT>
FORCE_INLINE void sfpu_activation_pack(uint32_t tile_index) {
    ActivationApplyHelper<ACT, 0, 0>::apply(tile_index);
}
