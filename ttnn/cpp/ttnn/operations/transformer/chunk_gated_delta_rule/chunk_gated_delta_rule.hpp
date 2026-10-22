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
 * Implements flash-linear-attention `chunk_gated_delta_rule` forward on-device: the sequence is
 * processed in chunks of chunk_size tokens; the state-independent per-chunk work (WY inverse,
 * decays, intra-chunk terms) is fanned across cores and the recurrent state [K, V] is carried
 * from chunk to chunk on-core, in fp32 at HiFi4. Matches FLA `naive_chunk_gated_delta_rule`
 * numerics. How the work is split over cores is the program_config's choice (below).
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
 * All inputs, including the optional tensors, must be device tensors in TILE layout: the op casts
 * dtypes (q/k/v -> bf16; g, beta, initial_state -> fp32) but never relayouts, and a ROW_MAJOR input
 * fails the device op's validation. eye/tril/ones/masks are taken all four or none (a partial set
 * is ignored and all four rebuilt, a host upload on every call); only their dtype and layout are
 * validated, not their shape against chunk_size. memory_config places the device op's outputs
 * (and, on the phased path, its seven DRAM intermediates); it is not passed to the token-major
 * post-processing.
 *
 * Returns:
 *   o           [B, T, HV, V]           (default; ROW_MAJOR)
 *               [B*HV, T, V]  TILE       (when output_head_major)
 *               fp32 on the fused and phased paths, bf16 on mono
 *   final_state [B, HV, K, V]  fp32      (present iff output_final_state)
 *
 * program_config: which device implementation runs and how it is laid out (chunk_gated_delta_rule_config.hpp):
 * ChunkGdnFusedProgramConfig (one program, NP producers -> NV receivers per head over the NoC),
 * ChunkGdnPhasedProgramConfig (prep -> DRAM -> scan, the bit-exact reference) or
 * ChunkGdnMonoProgramConfig (the single-kernel op). std::nullopt: the fused path with the cost
 * model's geometry when it fits this grid and is predicted to beat phased, else phased.
 *
 * wy_inverse: how each chunk's WY inverse T_inv = (I + N)^-1 is computed — ChunkGdnWyInverse::HORNER
 * (matrix engine, every architecture; the reference), FORWARD_SUBSTITUTION (one solve on the SFPU,
 * Blackhole and chunk_size == 32 only) or AUTO (FORWARD_SUBSTITUTION wherever supported, Horner elsewhere).
 * The mono program is Horner-only.
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
    ChunkGdnWyInverse wy_inverse = ChunkGdnWyInverse::AUTO,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& eye = std::nullopt,
    const std::optional<ttnn::Tensor>& tril = std::nullopt,
    const std::optional<ttnn::Tensor>& ones = std::nullopt,
    const std::optional<ttnn::Tensor>& masks = std::nullopt);

}  // namespace ttnn::transformer
