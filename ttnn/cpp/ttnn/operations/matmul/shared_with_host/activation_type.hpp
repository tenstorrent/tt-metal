// TODO(nuked-op matmul): minimal shared enum kept so kernel_lib/sfpu_activation_helpers.hpp
// compiles after the matmul op was nuked. Re-home into matmul when the op is recreated.
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::matmul {

enum class KernelActivation : uint32_t {
    NONE,
    GELU,
    TANH,
    SILU,
    RELU6,
    SIGMOID,
    HARDSIGMOID,
    HARDTANH,
    SELU,
    SOFTPLUS
};

}  // namespace ttnn::operations::matmul
