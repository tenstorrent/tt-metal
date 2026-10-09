// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// One routed expert's FFN without an activation: y = ((x @ w_gate) * (x @ w_up)) @ w_down, on a single node.
// x is (M, K), w_gate and w_up are (K, H), w_down is (H, K) and y is (M, K). All are bfloat16, TILE layout and
// DRAM interleaved.
Tensor routed_expert_ffn(
    const Tensor& x,
    const Tensor& w_gate,
    const Tensor& w_up,
    const Tensor& w_down,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
