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

// KernelActivation is the host/device shared enum; kept at global scope so kernels
// can reference it unqualified.
using ttnn::operations::matmul::KernelActivation;

namespace compute_kernel_lib {

/**
 * Bundles an activation kind with its compile-time parameters into one type so
 * that matmul_block / bias_add helpers can accept "the activation" as a single
 * template argument instead of a (KernelActivation, uint32_t, uint32_t, uint32_t)
 * tuple whose parameter slots are interpreted differently per activation.
 *
 * Use the named aliases below at hand-written call sites so the parameter meaning
 * is explicit (HardtanhActivation<low, high>, SeluActivation<alpha, lambda>, …).
 * For host-driven kernels that read activation + params from compile-time args,
 * wrap them as ActivationOp<activation_type, activation_param0, activation_param1,
 * activation_param2>.
 */
template <KernelActivation ACT, uint32_t P0 = 0, uint32_t P1 = 0, uint32_t P2 = 0>
struct ActivationOp {
    static constexpr KernelActivation activation = ACT;
    static constexpr uint32_t param0 = P0;
    static constexpr uint32_t param1 = P1;
    static constexpr uint32_t param2 = P2;
};

// Named aliases — param meaning is documented per activation. See
// ActivationApplyHelper below for the authoritative per-activation semantics.
using NoneActivation = ActivationOp<KernelActivation::NONE>;
using SiluActivation = ActivationOp<KernelActivation::SILU>;
using HardsigmoidActivation = ActivationOp<KernelActivation::HARDSIGMOID>;
// TanhActivation / GeluActivation: Fast=0 selects the accurate variant; non-zero selects fast.
template <uint32_t Fast = 0>
using TanhActivation = ActivationOp<KernelActivation::TANH, Fast>;
template <uint32_t Fast = 0>
using GeluActivation = ActivationOp<KernelActivation::GELU, Fast>;
// Relu6Activation: MaxBits is the max value as a uint32_t bit pattern of the float;
// 0 selects the default 6.0f.
template <uint32_t MaxBits = 0>
using Relu6Activation = ActivationOp<KernelActivation::RELU6, MaxBits>;
// SigmoidActivation: VecMode 1=R, 2=C, else RC. Fast non-zero enables fast approximation.
template <uint32_t VecMode = 0, uint32_t Fast = 0>
using SigmoidActivation = ActivationOp<KernelActivation::SIGMOID, VecMode, Fast>;
// HardtanhActivation<low_bits, high_bits>: low/high are uint32_t bit patterns of the floats.
template <uint32_t LowBits, uint32_t HighBits>
using HardtanhActivation = ActivationOp<KernelActivation::HARDTANH, LowBits, HighBits>;
// SeluActivation<alpha_bits, lambda_bits>: alpha/lambda are uint32_t bit patterns.
template <uint32_t AlphaBits, uint32_t LambdaBits>
using SeluActivation = ActivationOp<KernelActivation::SELU, AlphaBits, LambdaBits>;
// SoftplusActivation<beta_bits, threshold_bits, beta_reciprocal_bits>: float bit patterns.
// beta must be non-zero (validated via static_assert in ActivationApplyHelper).
template <uint32_t BetaBits, uint32_t ThresholdBits, uint32_t BetaReciprocalBits>
using SoftplusActivation = ActivationOp<KernelActivation::SOFTPLUS, BetaBits, ThresholdBits, BetaReciprocalBits>;

// Helper templates to select activation variants based on parameters.
//
// All three abstractions run on the PACKER thread (TRISC2): SFPU activation
// is fused into the pack stage so it overlaps with the math thread starting
// the next K-block. Math-thread post-compute hooks (PostComputeFn) run before
// tile_regs_commit and are unaffected.
template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0>
struct ActivationInitHelper {
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
    static_assert(
        ACT == KernelActivation::NONE || ACT == KernelActivation::SILU || ACT == KernelActivation::TANH ||
            ACT == KernelActivation::GELU || ACT == KernelActivation::RELU6 || ACT == KernelActivation::SIGMOID ||
            ACT == KernelActivation::HARDSIGMOID || ACT == KernelActivation::HARDTANH ||
            ACT == KernelActivation::SELU || ACT == KernelActivation::SOFTPLUS,
        "Unsupported KernelActivation type for fused activation apply");

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

// Bulk packer-thread activation. Wraps the math/pack semaphore wait, packer
// dest-offset flip, per-tile apply loop, and final SFPU stall — replaces a
// plain tile_regs_wait() at the same spot in the kernel pipeline.
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

}  // namespace compute_kernel_lib
