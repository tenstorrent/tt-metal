// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

namespace {
// Pick chunk_M_tiles for a given allocated M (tile rows). Factored out so both
// the standalone wrapper (M from x's shape) and the fused MoE composite (M =
// max_dispatched_tokens_per_expert / TILE, since x is the whole shared buffer)
// share the exact same heuristic.
uint32_t pick_chunk_m_tiles(uint32_t M_tiles_full) {
    constexpr uint32_t kGridY = 8;
    constexpr uint32_t kMinChunkMTiles = 16;  // per_core_M >= 2
    constexpr uint32_t kMaxChunkMTiles = 64;  // per_core_M <= 8 (L1 cap)
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
    return chunk_M_tiles;
}
}  // namespace

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
    const uint32_t M_tiles_full = x.padded_shape()[-2] / 32;
    const uint32_t chunk_M_tiles = pick_chunk_m_tiles(M_tiles_full);

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
        expert_region_offsets,
        /*fused_m_tiles=*/0);
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

    // Per-expert composite. The FFN runs in FUSED-EXTRACT + DIRECT-WRITE mode:
    // its reader reads this expert's tokens DIRECTLY out of the shared
    // dispatched buffer at the expert's region offset (folding what a separate
    // ttnn::extract op used to do — no per-expert temp buffer, no extra
    // DRAM round-trip), and its writer places the result back into the SAME
    // dispatched buffer at the same region offset (folding ttnn::insert). One
    // device program per expert; the (mutated) dispatched buffer is returned.
    //
    // In-place read+write to the shared buffer is safe within and across the
    // chunk loop: for a given chunk the reader reads x rows at the START of the
    // chunk (gate/up matmul input) and the writer writes the SAME rows at the
    // END (down matmul output), so the read always precedes the write by a
    // compute data-dependency; each row is written exactly once, by its own
    // chunk's writer, after its own chunk's read. Different experts touch
    // disjoint (non-overlapping) regions. Rows the writer never touches
    // (tile-aligned slack, zero-count experts, buffer tail) keep their original
    // dispatched-buffer contents, which downstream `combine` never reads
    // (bounded per expert to [offset, offset + ceil_tile(count))).
    //
    // x is the whole shared dispatched buffer; the FFN's M_tiles_full (chunk /
    // loop math) is driven by fused_m_tiles = max_dispatched_tokens_per_expert
    // / TILE, NOT by x's (full-buffer) row count. The chunk_M_tiles heuristic
    // therefore matches what the old per-expert (max_tokens, emb) tensor used.
    constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;
    const uint32_t fused_m_tiles = (max_dispatched_tokens_per_expert + TILE - 1) / TILE;
    const uint32_t chunk_M_tiles = pick_chunk_m_tiles(fused_m_tiles);
    const std::optional<ttnn::DeviceComputeKernelConfig> ckc =
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt;
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        // Fused-extract direct-write: x == output == dispatched_buffer, with
        // expert_region_offsets so BOTH the reader (input slice) and the writer
        // (output slice) offset into this expert's region of that same buffer.
        // The op mutates dispatched_buffer in place; its return value is unused.
        ttnn::prim::unified_routed_expert_ffn(
            dispatched_buffer,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            chunk_M_tiles,
            ckc,
            dispatched_buffer,
            expert_region_offsets,
            fused_m_tiles);
    }
    return dispatched_buffer;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
