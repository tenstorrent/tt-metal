// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/extract/extract.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output,
    const std::optional<ttnn::Tensor>& expert_region_offsets) {
    // Single-op fused per-expert FFN. One device Program runs gate matmul,
    // up matmul, silu, multiply, down matmul as four phases inside the same
    // kernel. The kernel reads counts[global_expert_idx_table[local_expert_id]]
    // device-side at entry and bounds its chunk loop to
    // ceil(count / chunk_M_tiles) — chunks past the actual token count are
    // skipped entirely (no matmul, no mcast).
    //
    // chunk_M_tiles: any value in {16, 24, 32, 40, 48, 56, 64} (per_core_M =
    // chunk_M_tiles / GRID_Y, must be >= 2 and <= 8). M_tiles_full does NOT
    // need to be a multiple of chunk_M_tiles — the kernel runs
    // ceil(M_tiles_full / chunk_M_tiles) chunks, reader zero-fills L1 rows
    // past M_tiles_full in the last chunk, writer skips OOB output writes.
    //
    // Picker minimizes the number of chunks first (= ceil(M / chunk)). Each
    // chunk pays ~25 K-block handshakes (14 gate/up + 11 down) regardless of
    // per_core_M, so fewer chunks wins more than waste hurts compute. On tie
    // (same num_chunks), prefers smaller waste = closer to-aligned. For
    // DS-V3 this picks 64 for nearly all sizes; 5k → 64 (3 chunks, was 40
    // with 4 chunks); 25k → 64 (13 chunks, was 40 with 20 chunks).
    constexpr uint32_t kGridY = 8;
    constexpr uint32_t kMinChunkMTiles = 16;  // per_core_M >= 2
    constexpr uint32_t kMaxChunkMTiles = 64;  // per_core_M <= 8 (L1 cap)
    const uint32_t M_tiles_full = x.padded_shape()[-2] / 32;
    uint32_t chunk_M_tiles = kMaxChunkMTiles;
    uint32_t best_num_chunks = (M_tiles_full + kMinChunkMTiles - 1) / kMinChunkMTiles + 1;
    uint32_t best_waste = kMaxChunkMTiles + 1;
    for (uint32_t cand = kMinChunkMTiles; cand <= kMaxChunkMTiles; cand += kGridY) {
        const uint32_t num_chunks = (M_tiles_full + cand - 1) / cand;
        const uint32_t rem = M_tiles_full % cand;
        const uint32_t waste = (rem == 0) ? 0 : (cand - rem);
        const bool better = (num_chunks < best_num_chunks) || (num_chunks == best_num_chunks && waste < best_waste);
        if (better) {
            best_num_chunks = num_chunks;
            best_waste = waste;
            chunk_M_tiles = cand;
        }
    }

    return ttnn::prim::unified_routed_expert_ffn(
        x,
        gate_proj,
        up_proj,
        down_proj,
        counts,
        global_expert_idx_table,
        local_expert_id,
        chunk_M_tiles,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        output,
        expert_region_offsets);
}

ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    // Per-expert composite: extract this expert's tokens out of the shared
    // dispatched buffer, run the unified FFN on them, and have the FFN's
    // writer place the result DIRECTLY into the shared output buffer at the
    // expert's region offset (direct-write mode). This fuses what used to be
    // a separate ttnn::insert op into the FFN writer — the FFN no longer
    // writes a per-expert temp buffer that insert then copies into the shared
    // buffer; it writes the shared buffer once. Same loop applies regardless
    // of `num_routed_experts`.
    //
    // `tokens` from extract is a per-expert (max_dispatched_tokens_per_expert,
    // emb) tensor with rows starting at 0. The FFN reads from row 0 of its
    // inputs; passing expert_region_offsets makes the writer add
    // expert_region_offsets[global_expert_id]/TILE tile-rows so the output
    // lands in this expert's slice of `expert_outputs`.
    //
    // Zero-initialized (not ttnn::empty): the FFN writer only writes each
    // expert's valid token rows, so padding rows — tile-aligned slack within a
    // region, regions of zero-count experts, and the tail of the buffer —
    // would otherwise keep uninitialized DRAM garbage (incl. NaN/Inf bit
    // patterns). The torch reference zeros these, and a NaN in padding would
    // corrupt any downstream masked reduction/combine, so the buffer is zeroed
    // up front.
    //
    // zeros_like (not zeros): dispatched_buffer is a TILE device tensor in a
    // device-fill-eligible dtype (bf8/bf16/fp32), so zeros_like takes the
    // on-device ttnn::fill path — no host-side std::vector(volume) + H2D copy
    // (which plain ttnn::zeros would incur for a large dispatch buffer) and no
    // device-pointer deref here.
    auto expert_outputs = ttnn::zeros_like(
        dispatched_buffer,
        /*dtype=*/std::nullopt,
        /*layout=*/std::nullopt,
        /*device=*/std::nullopt,
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        auto tokens = ttnn::extract(
            dispatched_buffer,
            expert_region_offsets,
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            max_dispatched_tokens_per_expert);
        // Direct-write: output == expert_outputs (the shared buffer),
        // expert_region_offsets supplied so the writer offsets into this
        // expert's region. Returns the same expert_outputs handle.
        expert_outputs = unified_routed_expert_ffn(
            tokens,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            compute_kernel_config,
            expert_outputs,
            expert_region_offsets);
    }
    return expert_outputs;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
