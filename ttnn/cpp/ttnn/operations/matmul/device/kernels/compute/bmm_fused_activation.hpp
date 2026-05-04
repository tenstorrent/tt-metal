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
            tanh_tile_init_pack<PARAM0 != 0>();
        } else if constexpr (ACT == KernelActivation::GELU) {
            // PARAM0: 0 = accurate, non-zero = fast
            gelu_tile_init_pack<PARAM0 != 0>();
        } else if constexpr (ACT == KernelActivation::RELU6) {
            relu_max_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::SIGMOID) {
            // Enhanced: PARAM1 is fast_approximate flag
            sigmoid_tile_init_pack<PARAM1 != 0>();
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

template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0, uint32_t PARAM2 = 0>
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
        ACT != KernelActivation::SOFTPLUS || PARAM0 != 0,
        "SOFTPLUS PARAM0 (beta) must be non-zero to avoid division by zero");

    FORCE_INLINE static void apply(uint32_t tile_index) {
        if constexpr (ACT == KernelActivation::SILU) {
            silu_tile_pack(tile_index);
        } else if constexpr (ACT == KernelActivation::TANH) {
            // PARAM0: 0 = accurate, non-zero = fast
            tanh_tile_pack<PARAM0 != 0>(tile_index);
        } else if constexpr (ACT == KernelActivation::GELU) {
            // PARAM0: 0 = accurate, non-zero = fast
            gelu_tile_pack<PARAM0 != 0>(tile_index);
        } else if constexpr (ACT == KernelActivation::RELU6) {
            // PARAM0 is the max value (as uint32_t bit pattern)
            // Default to 6.0 if PARAM0 is 0
            constexpr uint32_t max = (PARAM0 != 0) ? PARAM0 : 0x40c00000u;
            relu_max_tile_pack(tile_index, max);
        } else if constexpr (ACT == KernelActivation::SIGMOID) {
            // Enhanced: PARAM0 is vector mode, PARAM1 is fast_approximate
            constexpr int vec_mode = (PARAM0 == 1) ? VectorMode::R : (PARAM0 == 2) ? VectorMode::C : VectorMode::RC;
            sigmoid_tile_pack<vec_mode, PARAM1 != 0>(tile_index);
        } else if constexpr (ACT == KernelActivation::HARDSIGMOID) {
            hardsigmoid_tile_pack(tile_index);
        } else if constexpr (ACT == KernelActivation::HARDTANH) {
            hardtanh_tile_pack(tile_index, PARAM0, PARAM1);
        } else if constexpr (ACT == KernelActivation::SELU) {
            // PARAM0 is alpha, PARAM1 is lambda
            selu_tile_pack(tile_index, PARAM0, PARAM1);
        } else if constexpr (ACT == KernelActivation::SOFTPLUS) {
            // PARAM0 is beta, PARAM2 beta reciprocal, PARAM1 is threshold
            softplus_tile_pack(tile_index, PARAM0, PARAM2, PARAM1);
        }
    }
};

template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0, uint32_t PARAM2 = 0>
FORCE_INLINE void apply_activation_from_pack(uint32_t out_subblock_num_tiles) {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));

    // Flip destination register offset for PACKER access
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        ActivationApplyHelper<ACT, PARAM0, PARAM1, PARAM2>::apply(i);
    }

    // Wait for SFPU completion before packing
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}
