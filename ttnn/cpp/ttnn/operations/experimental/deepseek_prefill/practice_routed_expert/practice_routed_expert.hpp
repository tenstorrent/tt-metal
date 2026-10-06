// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert {

// One Kimi K3 routed expert as a single fused op, on that expert's tokens only:
//   out = situ_glu(x @ w_gate, x @ w_up) @ w_down
// with SiTU-GLU's betas (4 for gate, 25 for up) baked into the kernel. A practice version of
// unified_routed_expert_moe without its dispatch buffer, token counts, region offsets or expert-id table.
//
// Args:
//   x: (T, K) TILE BFLOAT16 or BFLOAT8_B, DRAM interleaved.
//   w_gate, w_up: (K, N) TILE, DRAM interleaved, transposed from the HF (out, in) layout.
//   w_down: (N, K), same layout and dtype as w_gate.
//   compute_kernel_config: matmul math fidelity and accumulator. Defaults to HiFi4 with fp32 accumulation.
//
// Returns:
//   ttnn::Tensor: (T, K) TILE in x's dtype, DRAM interleaved.
ttnn::Tensor practice_routed_expert(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert

namespace ttnn {
using operations::experimental::deepseek_prefill::practice_routed_expert::practice_routed_expert;
}  // namespace ttnn
