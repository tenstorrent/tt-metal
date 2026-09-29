// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "chunk_gated_delta_rule_config.hpp"

namespace ttnn::transformer {

/**
 * Standalone chunked Gated Delta Rule forward (from scratch, FLA algorithm).
 *
 * Implements flash-linear-attention `chunk_gated_delta_rule` forward on-device:
 * one Tensix core per (B*HV) head, sequential over chunks, holding the recurrent
 * state on-core. Matches FLA `naive_chunk_gated_delta_rule` numerics (fp32/HiFi4).
 *
 *   q    [B, T, H,  K]   or flat [B, T, H*K]
 *   k    [B, T, H,  K]   or flat [B, T, H*K]
 *   v    [B, T, HV, V]   or flat [B, T, HV*V]
 *   g    [B, T, HV]      log-space decay
 *   beta [B, T, HV]
 *
 * The rank of q/k/v selects the input path (no flag):
 *   rank 4, head-split: L2-normalized q/k expected (use_qk_l2norm stays false); the host applies
 *     scale, casts to bf16, permutes head-major, expands H -> HV for GQA and pads T to the chunk.
 *   rank 3, flat token-major: raw (unnormalized, unscaled) per-head concatenations as a projection or
 *     the causal conv emits them; no host relayout — the prep reader addresses each head's chunk out of
 *     the flat grid, maps value heads to key heads and applies the L2 norm and scale in-kernel.
 *     Requires K == V (H = flat q width / V, HV from beta), chunk_size == 32 and T % chunk_size == 0;
 *     fused and phased paths only.
 *
 * Returns:
 *   o           [B, T, HV, V]           (default; ROW_MAJOR)
 *               [B*HV, T, V]  TILE       (when output_head_major)
 *   final_state [B, HV, K, V]  (present iff output_final_state)
 *
 * program_config: which device implementation runs and how it is laid out (chunk_gated_delta_rule_config.hpp):
 * ChunkGdnFusedProgramConfig (one program, NP producers -> NV receivers per head over the NoC),
 * ChunkGdnPhasedProgramConfig (prep -> DRAM -> scan, the bit-exact reference) or
 * ChunkGdnMonoProgramConfig (the single-kernel op). std::nullopt: the fused path with the cost
 * model's geometry when it fits this grid and is predicted to beat phased, else phased. All three
 * paths are bit-identical for the same inputs and compute_kernel_config.
 *
 * output_head_major: the kernel natively produces o head-major ([BH,T,V]); the default
 * path permutes it to token-major [B,T,HV,V]. Callers that want head-major (e.g. the qwen36
 * GDN adapter's return_o_bh) should set this to get [BH,T,V] TILE directly and skip a
 * token<->head permute round-trip on both sides.
 */
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_gated_delta_rule(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    uint32_t chunk_size = 64,
    bool use_qk_l2norm = false,
    bool output_head_major = false,
    const std::optional<ChunkGdnProgramConfig>& program_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& eye = std::nullopt,
    const std::optional<ttnn::Tensor>& tril = std::nullopt,
    const std::optional<ttnn::Tensor>& ones = std::nullopt,
    const std::optional<ttnn::Tensor>& masks = std::nullopt);

}  // namespace ttnn::transformer
