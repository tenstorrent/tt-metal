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
    bool x_is_row_major,
    RoutedExpertActivation activation,
    const std::optional<uint32_t>& chunk_sizing_m_tiles) {
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
    // Picker minimizes the number of chunks first (= ceil(target / chunk)) over
    // the SIZING TARGET below. Each chunk pays ~25 K-block handshakes (14 gate/up
    // + 11 down) regardless of per_core_M, so fewer chunks wins more than waste
    // hurts compute. On tie (same num_chunks), prefers smaller waste (closer to
    // chunk-aligned). e.g. target 160 tiles (5,120 tokens) -> 56 (3 chunks, waste
    // 8); target 64 tiles (2,048 tokens) -> 64 (1 chunk, waste 0).
    constexpr uint32_t kGridY = 8;
    constexpr uint32_t kMinChunkMTiles = 16;  // per_core_M >= 2
    constexpr uint32_t kMaxChunkMTiles = 64;  // per_core_M <= 8 (L1 cap)
    // This expert's M in tiles. Defaults to x's allocated M; a caller passing a
    // shared x buffer (wider than one region) supplies the per-expert value.
    const uint32_t M_tiles_full = input_m_tiles.value_or(x.padded_shape()[-2] / 32);
    // Size chunk_M_tiles (per_core_M = chunk_M_tiles / GRID_Y) to the EXPECTED
    // active token count, not the allocated buffer M_tiles_full. An over-
    // allocated / under-filled buffer otherwise makes the picker choose a larger
    // chunk_M_tiles and pay phantom per_core_M work on every running chunk. Safe:
    // the device-side count still bounds effective_chunks and M_tiles_full still
    // bounds every OOB guard, so a too-small target only adds chunks (never drops
    // tokens). Defaults to M_tiles_full => behavior unchanged when not supplied.
    uint32_t sizing_m_tiles = chunk_sizing_m_tiles.value_or(M_tiles_full);
    if (sizing_m_tiles < 1) {
        sizing_m_tiles = 1;
    }
    if (sizing_m_tiles > M_tiles_full) {
        sizing_m_tiles = M_tiles_full;
    }
    uint32_t chunk_M_tiles = kMaxChunkMTiles;
    uint32_t best_num_chunks = (sizing_m_tiles + kMinChunkMTiles - 1) / kMinChunkMTiles + 1;
    uint32_t best_waste = kMaxChunkMTiles + 1;
    for (uint32_t cand = kMinChunkMTiles; cand <= kMaxChunkMTiles; cand += kGridY) {
        const uint32_t num_chunks = (sizing_m_tiles + cand - 1) / cand;
        const uint32_t rem = sizing_m_tiles % cand;
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
        x_is_row_major,
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
    RoutedExpertActivation activation,
    const std::optional<uint32_t>& expected_tokens_per_expert) {
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    // Per-expert composite: run the unified FFN on each expert's slice of the
    // dispatched buffer at that expert's region offset (read_x_at_offset for the
    // reader, expert_region_offsets for the writer). This fuses the old
    // ttnn::extract (input slice) + ttnn::insert (output placement) pair into the
    // FFN's reader and writer — no per-expert temp buffer, no extra DRAM round
    // trip. Same loop regardless of `num_routed_experts`.
    //
    // x is the whole shared buffer, so pass this expert's row count
    // (max_dispatched_tokens_per_expert in tiles) as input_m_tiles — the op sizes
    // its grid/chunks to one expert, not the buffer.
    //
    // The input layout selects the output strategy:
    //   * TILE buffer -> write IN PLACE (output == dispatched_buffer). The reader
    //     reads x in phase 1 and the writer drains cb_out only after compute
    //     consumes it, so a row's write is ordered after its read via the CB
    //     chain; chunks cover disjoint rows and experts touch disjoint regions,
    //     so no expert can disturb another. No allocation, no up-front fill.
    //   * ROW_MAJOR bf16 buffer -> the FFN tilizes x and packs bf8 internally, so
    //     input and output differ in both layout and dtype and cannot alias. One
    //     shared TILE bf8 output is allocated once for all experts; each writes
    //     its own region. Left uninitialized (no fill): downstream `combine`
    //     reads only written rows (bounded per expert to
    //     [offset, offset + ceil_tile(count))).
    const bool x_is_row_major = dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    const ttnn::Tensor output =
        x_is_row_major ? ttnn::empty(
                             dispatched_buffer.logical_shape(),
                             tt::tt_metal::DataType::BFLOAT8_B,
                             tt::tt_metal::Layout::TILE,
                             dispatched_buffer.device(),
                             tt::tt_metal::MemoryConfig{
                                 tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM})
                       : dispatched_buffer;
    const uint32_t m_tiles = (max_dispatched_tokens_per_expert + 31) / 32;
    // Chunk-sizing target: the expected active count (in tiles), if the caller
    // supplied it, else the capacity m_tiles (unchanged). Only affects the
    // chunk_M_tiles / per_core_M pick; every correctness bound still uses m_tiles.
    const std::optional<uint32_t> chunk_sizing_m_tiles =
        expected_tokens_per_expert.has_value() ? std::optional<uint32_t>((expected_tokens_per_expert.value() + 31) / 32)
                                               : std::nullopt;
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        unified_routed_expert_ffn(
            dispatched_buffer,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            compute_kernel_config,
            output,
            expert_region_offsets,
            m_tiles,
            /*read_x_at_offset=*/true,
            x_is_row_major,
            activation,
            chunk_sizing_m_tiles);
    }
    return output;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
