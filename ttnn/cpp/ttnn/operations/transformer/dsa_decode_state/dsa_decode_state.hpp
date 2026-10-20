// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::transformer {

/**
 * Fused DSA decode state maintenance (one program): from the decode stem matmul output, the q_a RMSNorm
 * (returned as q_resid) and, per user, the latent append (RMSNorm'd kv_a row into its paged latent cache),
 * the indexer ring update (LayerNorm'd k row and raw gate row) and the current pool's pooled key (the
 * softmax-weighted mean over the ring) into the pool cache -- all four caches updated IN PLACE.
 * See the Python binding for the tensor contract.
 */
ttnn::Tensor dsa_decode_state(
    const ttnn::Tensor& stem,
    const ttnn::Tensor& latent_cache,
    const ttnn::Tensor& latent_page_table,
    const ttnn::Tensor& positions,
    const ttnn::Tensor& ring_k,
    const ttnn::Tensor& ring_g,
    const ttnn::Tensor& pool_cache,
    const ttnn::Tensor& qa_norm_w,
    const ttnn::Tensor& kva_norm_w,
    const ttnn::Tensor& k_norm_w,
    const ttnn::Tensor& k_norm_b,
    const ttnn::Tensor& ape,
    const ttnn::Tensor& selectors,
    uint32_t qa_offset,
    uint32_t kva_offset,
    uint32_t k_offset,
    uint32_t g_offset,
    uint32_t q_lora_rank,
    uint32_t latent_dim,
    uint32_t index_dim,
    uint32_t pool_size,
    uint32_t pool_blocks,
    float eps_rms = 1e-5f,
    float eps_ln = 1e-6f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
