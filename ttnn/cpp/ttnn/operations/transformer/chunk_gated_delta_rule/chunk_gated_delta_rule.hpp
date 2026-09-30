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
 *   q    [B, T, H,  K]
 *   k    [B, T, H,  K]
 *   v    [B, T, HV, V]
 *   g    [B, T, HV]      log-space decay
 *   beta [B, T, HV]
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
 * wy_inverse: how each chunk's WY inverse T_inv = (I + N)^-1 is computed — ChunkGdnWyInverse::HORNER
 * (matrix engine, every architecture; the reference), SFPU (one forward-substitution solve on the
 * SFPU, Blackhole and chunk_size == 32 only) or AUTO (SFPU wherever supported, Horner elsewhere).
 * Unlike program_config this changes the arithmetic (PCC-class between methods); for a given method
 * every path is still bit-identical.
 *
 * output_head_major: the kernel natively produces o head-major ([BH,T,V]); the default
 * path permutes it to token-major [B,T,HV,V]. Callers that want head-major (e.g. the qwen36
 * GDN adapter's return_o_bh) should set this to get [BH,T,V] TILE directly and skip a
 * token<->head permute round-trip on both sides.
 *
 * final_state_output: an optional pre-allocated final-state tensor (fp32 TILE, interleaved,
 * [B, HV, K, V] or [B*HV, K, V]). When given, the kernel writes the final state straight into it
 * (no new state tensor is allocated) and the op returns this tensor as final_state. It may be the
 * initial_state tensor itself (in-place state update): each scan core reads its own state slice
 * before it writes the same slice. Requires output_final_state; fused and phased paths only.
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
    const std::optional<ttnn::Tensor>& masks = std::nullopt,
    // gb_flat (Option B): passing `sel` enables it (see chunk_gated_delta_rule.cpp). g/beta are then
    // read straight from the model's [B,T,HV] fp32 tensor by the prep reader, skipping the
    // headvec_split_tile permute+reshape. `sel` is the [1,1,32,32*HV] fp32 TILE one-hot head
    // selector (tile h picks head h's column). Fused path only. Build it once on the model/layer
    // (device-resident before trace capture, like eye/tril/ones/masks).
    const std::optional<ttnn::Tensor>& sel = std::nullopt,
    // qk_prenormed: q/k arrive already L2-normalized per head over K, q also multiplied by `scale`
    // (k / sqrt(sum k^2 + 1e-6), the in-kernel norm's formula), as flat [B,T,H*K] BFLOAT16 or FLOAT32 TILE; the
    // in-kernel norm is skipped, and FLOAT32 q/k are consumed as fp32 (no bf16 cast). Flat q/k, chunk_size 32, fused
    // path only.
    bool qk_prenormed = false,
    // decay_sfpu: the producer's per-chunk decay chain (decay, exp(decay), exp(g_sum - decay), the decay mask L and
    // dl*I) runs as two fp32 SFPU passes in DST instead of ~13 single-tile FPU ops: faster, and more accurate (fp32
    // SFPU arithmetic instead of tf32 FPU operands), so it changes bits. Fused path and chunk_size 32 only.
    bool decay_sfpu = false,
    const std::optional<ttnn::Tensor>& final_state_output = std::nullopt);

}  // namespace ttnn::transformer
