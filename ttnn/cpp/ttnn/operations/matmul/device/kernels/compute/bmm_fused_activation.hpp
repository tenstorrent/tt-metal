// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "internal/risc_attribs.h"

template <KernelActivation ACT>
FORCE_INLINE void init_sfpu_activation_pack() {
    if constexpr (ACT == KernelActivation::SILU) {
        return silu_tile_init_pack();
    } else if (ACT == KernelActivation::TANH) {
        return tanh_tile_init_pack();
    } else if (ACT == KernelActivation::GELU) {
        return gelu_tile_init_pack();
    }
}

template <KernelActivation ACT>
FORCE_INLINE void sfpu_activation_pack(uint32_t tile_index) {
    if constexpr (ACT == KernelActivation::SILU) {
        return silu_tile_pack(tile_index);
    } else if (ACT == KernelActivation::TANH) {
        return tanh_tile_pack(tile_index);
    } else if (ACT == KernelActivation::GELU) {
        return gelu_tile_pack(tile_index);
    }
}
