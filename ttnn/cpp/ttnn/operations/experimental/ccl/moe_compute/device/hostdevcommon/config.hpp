// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::experimental::prim::detail {

// Activation function types for MoE operations
enum class MoEActivationFunction : uint8_t { SILU = 0, SWIGLU = 1 };

// Configuration type for selecting MoE architecture
enum class MoEConfigType : uint32_t { DEEPSEEK = 0, GPT = 1 };

}  // namespace ttnn::experimental::prim::detail
