// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
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
    // This removes the previous host-side pad/slice round-trip.
    //
    // Picker minimizes last-chunk waste (chunk_M_tiles - M_tiles_full %
    // chunk_M_tiles, or 0 if cleanly divides). On tie, prefer larger
    // chunk_M_tiles for fewer chunk-loop iterations and less per-chunk
    // overhead. For DS-V3 dims this picks 64 for clean cases (2k, 4k, 8k,
    // 16k), 40 for 25k, 56 for 1.6k / 12345-token cases — minimizing the
    // number of zero-padded tile rows the last chunk has to process.
    constexpr uint32_t kGridY = 8;
    constexpr uint32_t kMinChunkMTiles = 16;  // per_core_M >= 2
    constexpr uint32_t kMaxChunkMTiles = 64;  // per_core_M <= 8 (L1 cap)
    const uint32_t M_tiles_full = x.padded_shape()[-2] / 32;
    uint32_t chunk_M_tiles = kMinChunkMTiles;
    uint32_t best_waste = kMaxChunkMTiles + 1;
    for (uint32_t cand = kMinChunkMTiles; cand <= kMaxChunkMTiles; cand += kGridY) {
        const uint32_t rem = M_tiles_full % cand;
        const uint32_t waste = (rem == 0) ? 0 : (cand - rem);
        if (waste <= best_waste) {
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

    // The per-expert loop runs entirely C++-side. For each local expert we:
    //   1. extract its tokens out of the dispatched buffer (returns the full
    //      max_dispatched_tokens_per_expert-sized buffer; the first count
    //      rows hold this expert's valid tokens, the rest is dispatch
    //      padding).
    //   2. run the unified routed-expert FFN. The op reads counts and the
    //      local->global idx mapping on-device and bounds its chunk loop to
    //      ceil(count / chunk_M_tiles) — no host-side sync, no per-expert
    //      narrow.
    //   3. insert the FFN output back into the expert_outputs buffer at this
    //      expert's region.
    // No host-side pad/slice. The unified FFN kernel handles arbitrary
    // M_tiles_full — chunk_M_tiles is picked to minimize last-chunk waste,
    // and the kernel's reader zero-fills L1 for tile rows past
    // min(count_tiles, M_tiles_full), writer skips OOB output writes. So the
    // extracted tokens flow straight into the FFN with no extra allocation
    // or copy, and the FFN output goes straight into insert.
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
            local_expert,
            compute_kernel_config,
            std::nullopt);

        expert_outputs = ttnn::insert(
            expert_outputs, ffn_out, expert_region_offsets, expert_token_counts, global_expert_idx_table, local_expert);
    }

    return expert_outputs;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
