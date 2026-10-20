// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::transformer {

/**
 * Fused KDA (Kimi Delta Attention) single-token decode block, everything between the packed
 * projection matmuls and o_proj in ONE program: causal conv (kernel 4) over the device-owned
 * history + SiLU, per-head L2 norms, the per-key-channel forget gate, beta and output gates, the
 * gated delta-rule recurrence on the device-owned state and the gated RMSNorm. The recurrent state
 * and the conv history are updated IN PLACE; the output is the packed [1, 1, B, H*V] bf16 ROW_MAJOR
 * tensor (tilized, it feeds o_proj). See the Python binding for the tensor contract.
 */
ttnn::Tensor kda_decode(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& gates,
    const ttnn::Tensor& conv_state,
    const ttnn::Tensor& recurrent_state,
    const ttnn::Tensor& taps,
    const ttnn::Tensor& chan_consts,
    const ttnn::Tensor& o_norm_weight,
    const ttnn::Tensor& selectors,
    uint32_t num_heads,
    float gate_lower_bound = -5.0f,
    float l2_eps = 1e-6f,
    float norm_eps = 1e-5f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
