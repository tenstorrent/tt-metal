// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::quasar::matmul {

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

}  // namespace ttnn::operations::experimental::quasar::matmul
