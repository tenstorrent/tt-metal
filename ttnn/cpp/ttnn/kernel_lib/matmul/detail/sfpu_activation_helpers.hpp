// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/activation_types.hpp"
#include "internal/risc_attribs.h"

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/mish.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/elu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

// Keep direct calls and compile-time thread selection without repeating each branch.
// Parenthesize call expressions whose template arguments contain commas.
#define TTNN_SFPU_DISPATCH(thread, math_call, pack_call)                          \
    do {                                                                          \
        if constexpr ((thread) == ::compute_kernel_lib::ActivationThread::Math) { \
            math_call;                                                            \
        } else {                                                                  \
            pack_call;                                                            \
        }                                                                         \
    } while (false)

namespace compute_kernel_lib::detail {

// Initialize once at kernel startup on the selected thread.
template <
    KernelActivation Act,
    uint32_t Param0 = 0,
    uint32_t Param1 = 0,
    ActivationThread Thread = ActivationThread::Pack>
struct ActivationInitHelper {
    FORCE_INLINE static void init();
};

template <KernelActivation Act>
inline constexpr bool is_supported_activation =
    Act == KernelActivation::NONE || Act == KernelActivation::SILU || Act == KernelActivation::TANH ||
    Act == KernelActivation::GELU || Act == KernelActivation::GELU_TANH || Act == KernelActivation::RELU6 ||
    Act == KernelActivation::SIGMOID || Act == KernelActivation::HARDSIGMOID || Act == KernelActivation::HARDTANH ||
    Act == KernelActivation::SELU || Act == KernelActivation::SOFTPLUS || Act == KernelActivation::MISH ||
    Act == KernelActivation::SQRT || Act == KernelActivation::LEAKY_RELU || Act == KernelActivation::ELU ||
    Act == KernelActivation::EXP || Act == KernelActivation::RECIP;

template <KernelActivation Act, uint32_t Param0, uint32_t Param1, ActivationThread Thread>
FORCE_INLINE void ActivationInitHelper<Act, Param0, Param1, Thread>::init() {
    static_assert(is_supported_activation<Act>, "Unsupported KernelActivation type for fused activation init");

    if constexpr (Act == KernelActivation::MISH) {
        TTNN_SFPU_DISPATCH(Thread, mish_tile_init<(Param0 != 0)>(), mish_tile_init_pack<(Param0 != 0)>());
    } else if constexpr (Act == KernelActivation::SQRT) {
        TTNN_SFPU_DISPATCH(Thread, sqrt_tile_init(), sqrt_tile_init_pack());
    } else if constexpr (Act == KernelActivation::LEAKY_RELU) {
#ifndef ARCH_QUASAR
        TTNN_SFPU_DISPATCH(Thread, leaky_relu_tile_init(), leaky_relu_tile_init_pack());
#else
        static_assert(Act != KernelActivation::LEAKY_RELU, "LEAKY_RELU is unavailable on Quasar");
#endif
    } else if constexpr (Act == KernelActivation::ELU) {
#ifndef ARCH_QUASAR
        TTNN_SFPU_DISPATCH(Thread, elu_tile_init(), elu_tile_init_pack());
#else
        static_assert(Act != KernelActivation::ELU, "ELU is unavailable on Quasar");
#endif
    } else if constexpr (Act == KernelActivation::EXP) {
        TTNN_SFPU_DISPATCH(Thread, exp_tile_init<(Param0 != 0)>(), exp_packthread_tile_init<(Param0 != 0)>());
    } else if constexpr (Act == KernelActivation::RECIP) {
        TTNN_SFPU_DISPATCH(Thread, recip_tile_init(), recip_tile_init_pack());
    } else if constexpr (Act == KernelActivation::SILU) {
        TTNN_SFPU_DISPATCH(Thread, silu_tile_init(), silu_tile_init_pack());
    } else if constexpr (Act == KernelActivation::TANH) {
        TTNN_SFPU_DISPATCH(Thread, tanh_tile_init<(Param0 != 0)>(), tanh_tile_init_pack<(Param0 != 0)>());
    } else if constexpr (Act == KernelActivation::GELU) {
        TTNN_SFPU_DISPATCH(Thread, gelu_tile_init<(Param0 != 0)>(), gelu_tile_init_pack<(Param0 != 0)>());
    } else if constexpr (Act == KernelActivation::GELU_TANH) {
        TTNN_SFPU_DISPATCH(Thread, gelu_tanh_tile_init(), gelu_tanh_tile_init_pack());
    } else if constexpr (Act == KernelActivation::RELU6) {
        TTNN_SFPU_DISPATCH(Thread, relu_max_tile_init(), relu_max_tile_init_pack());
    } else if constexpr (Act == KernelActivation::SIGMOID) {
        TTNN_SFPU_DISPATCH(Thread, sigmoid_tile_init<(Param1 != 0)>(), sigmoid_tile_init_pack<(Param1 != 0)>());
    } else if constexpr (Act == KernelActivation::HARDSIGMOID) {
        TTNN_SFPU_DISPATCH(Thread, hardsigmoid_tile_init(), hardsigmoid_tile_init_pack());
    } else if constexpr (Act == KernelActivation::HARDTANH) {
        TTNN_SFPU_DISPATCH(Thread, hardtanh_tile_init(), hardtanh_tile_init_pack());
    } else if constexpr (Act == KernelActivation::SELU) {
        TTNN_SFPU_DISPATCH(Thread, selu_tile_init(), selu_tile_init_pack());
    } else if constexpr (Act == KernelActivation::SOFTPLUS) {
        TTNN_SFPU_DISPATCH(Thread, softplus_tile_init(), softplus_tile_init_pack());
    }
}

template <
    KernelActivation Act,
    uint32_t Param0 = 0,
    uint32_t Param1 = 0,
    uint32_t Param2 = 0,
    ActivationThread Thread = ActivationThread::Pack>
struct ActivationApplyHelper {
    static_assert(is_supported_activation<Act>, "Unsupported KernelActivation type for fused activation apply");

    static_assert(
        Act != KernelActivation::SOFTPLUS || Param0 != 0,
        "SOFTPLUS Param0 (beta) must be non-zero to avoid division by zero");

    FORCE_INLINE static void apply(uint32_t tile_index) {
        if constexpr (Act == KernelActivation::MISH) {
            TTNN_SFPU_DISPATCH(Thread, mish_tile<(Param0 != 0)>(tile_index), mish_tile_pack<(Param0 != 0)>(tile_index));
        } else if constexpr (Act == KernelActivation::SQRT) {
            TTNN_SFPU_DISPATCH(Thread, sqrt_tile<(Param0 != 0)>(tile_index), sqrt_tile_pack<(Param0 != 0)>(tile_index));
        } else if constexpr (Act == KernelActivation::LEAKY_RELU) {
#ifndef ARCH_QUASAR
            TTNN_SFPU_DISPATCH(Thread, leaky_relu_tile(tile_index, Param0), leaky_relu_tile_pack(tile_index, Param0));
#else
            static_assert(Act != KernelActivation::LEAKY_RELU, "LEAKY_RELU is unavailable on Quasar");
#endif
        } else if constexpr (Act == KernelActivation::ELU) {
#ifndef ARCH_QUASAR
            TTNN_SFPU_DISPATCH(Thread, elu_tile(tile_index, Param0), elu_tile_pack(tile_index, Param0));
#else
            static_assert(Act != KernelActivation::ELU, "ELU is unavailable on Quasar");
#endif
        } else if constexpr (Act == KernelActivation::EXP) {
            TTNN_SFPU_DISPATCH(
                Thread, exp_tile<(Param0 != 0)>(tile_index), exp_packthread_tile<(Param0 != 0)>(tile_index));
        } else if constexpr (Act == KernelActivation::RECIP) {
            TTNN_SFPU_DISPATCH(Thread, recip_tile(tile_index), recip_tile_pack(tile_index));
        } else if constexpr (Act == KernelActivation::SILU) {
            TTNN_SFPU_DISPATCH(Thread, silu_tile(tile_index), silu_tile_pack(tile_index));
        } else if constexpr (Act == KernelActivation::TANH) {
            TTNN_SFPU_DISPATCH(Thread, tanh_tile<(Param0 != 0)>(tile_index), tanh_tile_pack<(Param0 != 0)>(tile_index));
        } else if constexpr (Act == KernelActivation::GELU) {
            TTNN_SFPU_DISPATCH(Thread, gelu_tile<(Param0 != 0)>(tile_index), gelu_tile_pack<(Param0 != 0)>(tile_index));
        } else if constexpr (Act == KernelActivation::GELU_TANH) {
            TTNN_SFPU_DISPATCH(Thread, gelu_tanh_tile(tile_index), gelu_tanh_tile_pack(tile_index));
        } else if constexpr (Act == KernelActivation::RELU6) {
            constexpr uint32_t max = (Param0 != 0) ? Param0 : 0x40c00000u;
            TTNN_SFPU_DISPATCH(Thread, relu_max_tile(tile_index, max), relu_max_tile_pack(tile_index, max));
        } else if constexpr (Act == KernelActivation::SIGMOID) {
            constexpr VectorMode vec_mode = (Param0 == 1)   ? VectorMode::R
                                            : (Param0 == 2) ? VectorMode::C
                                                            : VectorMode::RC;
            TTNN_SFPU_DISPATCH(
                Thread,
                (sigmoid_tile<vec_mode, (Param1 != 0)>(tile_index)),
                (sigmoid_tile_pack<vec_mode, (Param1 != 0)>(tile_index)));
        } else if constexpr (Act == KernelActivation::HARDSIGMOID) {
            TTNN_SFPU_DISPATCH(Thread, hardsigmoid_tile(tile_index), hardsigmoid_tile_pack(tile_index));
        } else if constexpr (Act == KernelActivation::HARDTANH) {
            TTNN_SFPU_DISPATCH(
                Thread, hardtanh_tile(tile_index, Param0, Param1), hardtanh_tile_pack(tile_index, Param0, Param1));
        } else if constexpr (Act == KernelActivation::SELU) {
            TTNN_SFPU_DISPATCH(
                Thread, selu_tile(tile_index, Param0, Param1), selu_tile_pack(tile_index, Param0, Param1));
        } else if constexpr (Act == KernelActivation::SOFTPLUS) {
            // The SFPU API takes beta reciprocal before threshold.
            TTNN_SFPU_DISPATCH(
                Thread,
                softplus_tile(tile_index, Param0, Param2, Param1),
                softplus_tile_pack(tile_index, Param0, Param2, Param1));
        }
    }
};

// Apply activation to DST on the packer thread, replacing tile_regs_wait().
template <KernelActivation Act, uint32_t Param0 = 0, uint32_t Param1 = 0, uint32_t Param2 = 0>
FORCE_INLINE void apply_activation_from_pack(uint32_t out_subblock_num_tiles) {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));

    // Select the packer's DST half.
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        ActivationApplyHelper<Act, Param0, Param1, Param2, ActivationThread::Pack>::apply(i);
    }

    // Wait for SFPU completion before packing.
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

}  // namespace compute_kernel_lib::detail

#undef TTNN_SFPU_DISPATCH
