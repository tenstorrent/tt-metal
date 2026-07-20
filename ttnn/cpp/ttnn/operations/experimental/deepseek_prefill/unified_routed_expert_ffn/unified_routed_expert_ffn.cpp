// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/creation/creation.hpp"
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
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    const std::optional<uint32_t>& input_m_tiles,
    bool read_x_at_offset,
    RoutedExpertActivation activation) {
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
    // This expert's M in tiles. Defaults to x's allocated M; a caller passing a
    // shared x buffer (wider than one region) supplies the per-expert value.
    const uint32_t M_tiles_full = input_m_tiles.value_or(x.padded_shape()[-2] / 32);
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
        M_tiles_full,
        read_x_at_offset,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        output,
        expert_region_offsets,
        activation);
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
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    RoutedExpertActivation activation) {
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    // Per-expert composite: run the unified FFN on this expert's slice of the
    // dispatched buffer IN PLACE. The FFN reads x directly from the dispatched
    // buffer at the expert's region offset (read_x_at_offset) and its writer
    // places the result back into the SAME buffer at the same offset
    // (expert_region_offsets). This fuses what used to be a separate
    // ttnn::extract (input slice) + ttnn::insert (output placement) pair into
    // the FFN's reader and writer — no per-expert temp buffer, no extra DRAM
    // round-trip. Same loop regardless of `num_routed_experts`; the (mutated)
    // dispatched buffer is returned.
    //
    // x is the whole shared buffer, so pass this expert's row count
    // (max_dispatched_tokens_per_expert in tiles) as input_m_tiles — the op
    // sizes its grid/chunks to one expert, not the buffer.
    //
    // In-place read+write of one region is safe: within the op the reader reads
    // x in phase 1 and the writer drains cb_out only after compute consumes it,
    // so the write of a row is ordered after its read via the CB chain; chunks
    // cover disjoint rows. Across the loop each expert touches only its own
    // (non-overlapping) region, so the read/write of expert i cannot disturb
    // expert j's rows.
    //
    // No separate output allocation or zero-fill: the FFN writes back into the
    // existing dispatched buffer, so there is no per-call DRAM allocation and no
    // up-front fill. Rows the writer does not touch (tile-aligned slack within a
    // region, regions of zero-count experts, and the tail of the buffer) retain
    // their original contents, which downstream `combine` never reads (bounded
    // per expert to [offset, offset + ceil_tile(count))).
    const uint32_t m_tiles = (max_dispatched_tokens_per_expert + 31) / 32;
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        // output == dispatched_buffer with expert_region_offsets => writer
        // offsets into this expert's region; read_x_at_offset => reader reads x
        // from that same region. The op mutates dispatched_buffer in place; its
        // return value is unused (the composite returns dispatched_buffer below).
        unified_routed_expert_ffn(
            dispatched_buffer,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            compute_kernel_config,
            dispatched_buffer,
            expert_region_offsets,
            m_tiles,
            /*read_x_at_offset=*/true,
            activation);
    }
    return dispatched_buffer;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
