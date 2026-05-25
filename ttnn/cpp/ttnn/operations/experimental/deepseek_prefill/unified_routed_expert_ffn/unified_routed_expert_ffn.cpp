// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/extract/extract.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/insert/insert.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t local_expert_id,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output) {
    // Single-op fused routed-expert FFN. One device Program runs gate matmul,
    // up matmul, silu, multiply, down matmul as four phases inside the same
    // kernel. The kernel reads counts[global_expert_idx_table[local_expert_id]]
    // device-side at entry and bounds its chunk loop to
    // ceil(count / chunk_M_tiles) — chunks past the actual token count are
    // skipped entirely (no matmul, no mcast). All MANDATORY (per user):
    // device-side count sparsity, no host-side count read, no op2op gap from
    // per-chunk dispatch.
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
        expert_region_offsets,
        local_expert_id,
        chunk_M_tiles,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        output);
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

    // Path selection based on num_routed_experts. The fused extract+insert
    // path piggybacks expert_region_offsets onto CB_IDX_SCRATCH's L1 page
    // (page_size = idx_page + region_offsets_page). On the high-expert
    // configurations (num_routed_experts > 64, e.g. 256 experts with
    // experts_per_chip=32) the region_offsets page is large enough
    // (1024 bytes) to push the FFN's static CB region past a per-core L1
    // buffer placed by the L1 allocator, producing:
    //     "Statically allocated CBs ... clash with L1 buffers"
    // Falling back to the unfused extract -> FFN -> insert path for that
    // case avoids the issue (CB sizes shrink back to their pre-task-#37
    // values) at the cost of the extract+insert op2op gaps.
    const uint32_t num_routed_experts = static_cast<uint32_t>(expert_token_counts.logical_shape()[-1]);
    const bool use_fused_path = (num_routed_experts <= 64);

    if (use_fused_path) {
        // Fused extract + FFN + insert: the unified FFN kernel reads
        // region_offsets[global_expert_id] device-side and applies the
        // offset to DRAM tile indices for both x reads and output writes.
        // expert_outputs allocated with dispatched_buffer's shape/dtype;
        // each per-expert FFN call writes its [start_row,
        // start_row+ceil_tile(count)) slice. Slices non-overlapping per
        // expert_region_offsets / expert_token_counts contract.
        auto expert_outputs = ttnn::empty(
            dispatched_buffer.logical_shape(),
            dispatched_buffer.dtype(),
            ttnn::TILE_LAYOUT,
            dispatched_buffer.device(),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});

        for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
            (void)unified_routed_expert_ffn(
                dispatched_buffer,
                gate_projs[local_expert],
                up_projs[local_expert],
                down_projs[local_expert],
                expert_token_counts,
                global_expert_idx_table,
                expert_region_offsets,
                local_expert,
                compute_kernel_config,
                /*output=*/expert_outputs);
        }
        return expert_outputs;
    }

    // Fallback: unfused extract -> FFN -> insert per expert. Used for
    // num_routed_experts > 64 (e.g. 256 experts). Same semantics as the
    // pre-task-#37 path. expert_outputs aliases dispatched_buffer (insert
    // writes in-place); FFN gets a per-expert extracted buffer.
    auto expert_outputs = dispatched_buffer;
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        auto tokens = ttnn::extract(
            dispatched_buffer,
            expert_region_offsets,
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            max_dispatched_tokens_per_expert);
        auto ffn_out = unified_routed_expert_ffn(
            tokens,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            expert_region_offsets,
            local_expert,
            compute_kernel_config,
            std::nullopt);
        expert_outputs = ttnn::insert(
            expert_outputs, ffn_out, expert_region_offsets, expert_token_counts, global_expert_idx_table, local_expert);
    }
    return expert_outputs;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
