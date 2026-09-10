// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Fused CSA compressor pool (deepseek_v4_flash `DeepSeekV4CSACompressor._pool_window`).
//
// Replaces the Python sequence in attention.py:
//
//     prev_g = prev_gate + position_bias
//     cur_g  = win_gate  + position_bias
//     new_kv   = concat(prev_kv[..., :Dh], win_kv[..., Dh:], dim=window)
//     new_gate = concat(prev_g[..., :Dh],  cur_g[..., Dh:],  dim=window)
//     out = sum(softmax(new_gate, dim=window) * new_kv, dim=window)   # [1, 1, users, Dh]
//
// Every window tensor is ROW_MAJOR WIDTH_SHARDED over the last dim `2*Dh`. Ca lives on
// the first half of that grid and Cb on the second; each Ca core NoC-reads its Cb
// partner so the 2*cr softmax is local. `position_bias` is the same grid with height
// `compress_rate` (broadcast over users).
//
// Args:
//   prev_kv / prev_gate / win_kv / win_gate: [1, 1, users * compress_rate, 2*Dh]
//   position_bias: [1, 1, compress_rate, 2*Dh]
//
// Returns: [1, 1, users, Dh] ROW_MAJOR WIDTH_SHARDED on the Ca (first-half) cores.
Tensor csa_pool_window(
    const Tensor& prev_kv,
    const Tensor& prev_gate,
    const Tensor& win_kv,
    const Tensor& win_gate,
    const Tensor& position_bias,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek
