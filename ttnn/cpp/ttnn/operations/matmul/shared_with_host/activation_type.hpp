// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

enum class KernelActivation : uint32_t {
    NONE,
    GELU,
    TANH,
    SILU,
};
