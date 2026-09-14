// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// M0a SCRATCH: row-batched T-loop recurrence prototype (one core per (user, value head); state resident in L1
// across the T candidate tokens; per-token fp32 state written to ring block t*B*Nv + u*Nv + h). Not a product.
ttnn::Tensor gdn_spec_tloop_proto(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& neg_exp_A,
    const ttnn::Tensor& ring,
    const ttnn::Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t T,
    uint32_t B,
    uint32_t qkvz_dim,
    uint32_t s0_slot = 0,
    std::optional<float> scale = std::nullopt,
    float l2_epsilon = 1e-6f,
    float norm_epsilon = 1e-6f,
    bool row_batched = true,
    bool write_ring = true,
    uint32_t opt_flags = 0,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    ttnn::DataType output_dtype = ttnn::DataType::BFLOAT16);

}  // namespace ttnn::experimental::kda
