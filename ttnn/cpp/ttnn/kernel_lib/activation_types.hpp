// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace compute_kernel_lib {

enum class KernelActivation : uint32_t {
    NONE,
    GELU,
    GELU_TANH,
    TANH,
    SILU,
    RELU6,
    SIGMOID,
    HARDSIGMOID,
    HARDTANH,
    SELU,
    SOFTPLUS,
    MISH,
    SQRT,
    LEAKY_RELU,
    ELU,
    EXP,
    RECIP
};

enum class ActivationThread : uint32_t { Math, Pack };

}  // namespace compute_kernel_lib
