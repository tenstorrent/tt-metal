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

namespace {

// Picks chunk_M_tiles such that per_core_M = chunk_M_tiles / grid_y is in
// [2, per_core_M_max] and ceil(M_tiles_full / chunk) is minimized. On tie,
// prefers smaller waste (= padding past M_tiles_full).
uint32_t pick_chunk_M_tiles(uint32_t M_tiles_full, uint32_t grid_y, uint32_t per_core_M_max) {
    const uint32_t min_chunk = 2 * grid_y;
    const uint32_t max_chunk = per_core_M_max * grid_y;
    uint32_t best = max_chunk;
    uint32_t best_num_chunks = (M_tiles_full + min_chunk - 1) / min_chunk + 1;
    uint32_t best_waste = max_chunk + 1;
    for (uint32_t cand = min_chunk; cand <= max_chunk; cand += grid_y) {
        const uint32_t num_chunks = (M_tiles_full + cand - 1) / cand;
        const uint32_t rem = M_tiles_full % cand;
        const uint32_t waste = (rem == 0) ? 0 : (cand - rem);
        const bool better = (num_chunks < best_num_chunks) || (num_chunks == best_num_chunks && waste < best_waste);
        if (better) {
            best_num_chunks = num_chunks;
            best_waste = waste;
            best = cand;
        }
    }
    return best;
}

// Largest divisor of K_gate_tiles that is <= max_block. The WH variant uses
// this to choose in0_block_w_gu when K is not a multiple of 16 (e.g. 90 for
// emb=2880). Falls back to 1 if no divisor is found (always safe).
uint32_t pick_in0_block_w(uint32_t K_gate_tiles, uint32_t max_block) {
    for (uint32_t d = max_block; d >= 1; --d) {
        if (K_gate_tiles % d == 0) {
            return d;
        }
    }
    return 1;
}

}  // namespace

ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t local_expert_id,
    bool use_region_offsets,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output) {
    // Blackhole 11x8 = 88-core fused routed-expert FFN. See header for the
    // full math/contract. Picker uses GRID_Y = 8 and a max per_core_M of 8.
    constexpr uint32_t kGridX = 11;
    constexpr uint32_t kGridY = 8;
    constexpr uint32_t kIn0BlockWGu = 16;
    constexpr uint32_t kPerCoreMMax = 8;
    const uint32_t M_tiles_full = x.padded_shape()[-2] / 32;
    const uint32_t chunk_M_tiles = pick_chunk_M_tiles(M_tiles_full, kGridY, kPerCoreMMax);

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
        use_region_offsets,
        kGridX,
        kGridY,
        kIn0BlockWGu,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        output);
}

ttnn::Tensor unified_routed_expert_ffn_wh(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t local_expert_id,
    bool use_region_offsets,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output) {
    // Wormhole 8x8 = 64-core variant of the fused routed-expert FFN. The WH
    // worker grid is 8x8, so we cannot use the BH 11-column layout. The K
    // dim of the gate/up matmuls (K_gate_tiles = emb / TILE) is also no
    // longer required to be a multiple of 16 — we pick the largest divisor
    // of K_gate_tiles that fits L1 (capped at 10). per_core_M is capped at 4
    // because per-core L1 is tighter with the 8-wide N split.
    //
    // Why kIn0BlockWMax = 10 and not 16: the in1 (gate/up) double-buffered
    // mcast block has size in0_block_w_gu * per_core_N_gu * tile_size. With
    // the WH per_core_N_gu = 12 (vs BH's 6), in0_block_w_gu=15 (e.g. for
    // emb=2880, K_gate_tiles=90) produces a 180-tile block per CB slot and
    // the gate/up matmul outputs were observed to be non-deterministically
    // corrupt on the FIRST program invocation (PCC oscillating 0.93–0.98
    // across cold-cache runs, deterministic 0.98 after one warm-up).
    // Capping at 10 (in0_block_w_gu * per_core_N_gu = 120 tiles per slot)
    // gives stable PCC 0.98 on cold and warm runs and matches the in1 block
    // size envelope of the working BH 11x8 path (in0_block_w_gu=16 *
    // per_core_N_gu=6 = 96 tiles per slot).
    constexpr uint32_t kGridX = 8;
    constexpr uint32_t kGridY = 8;
    constexpr uint32_t kPerCoreMMax = 4;
    constexpr uint32_t kIn0BlockWMax = 10;
    const uint32_t K_gate_tiles = x.padded_shape()[-1] / 32;
    const uint32_t in0_block_w_gu = pick_in0_block_w(K_gate_tiles, kIn0BlockWMax);
    const uint32_t M_tiles_full = x.padded_shape()[-2] / 32;
    const uint32_t chunk_M_tiles = pick_chunk_M_tiles(M_tiles_full, kGridY, kPerCoreMMax);

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
        use_region_offsets,
        kGridX,
        kGridY,
        in0_block_w_gu,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        output);
}

namespace {

// Common MoE-level dispatch loop. Calls `ffn` (either the BH or WH variant)
// per expert. Identical structure to the prior unified_routed_expert_moe;
// extracted here so we don't duplicate the path-selection logic for the WH
// op.
template <typename FfnFn>
ttnn::Tensor unified_routed_expert_moe_impl(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    FfnFn&& ffn) {
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    const uint32_t num_routed_experts = static_cast<uint32_t>(expert_token_counts.logical_shape()[-1]);
    const bool use_fused_path = (num_routed_experts <= 64);

    if (use_fused_path) {
        auto expert_outputs = ttnn::empty(
            dispatched_buffer.logical_shape(),
            dispatched_buffer.dtype(),
            ttnn::TILE_LAYOUT,
            dispatched_buffer.device(),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});

        for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
            (void)ffn(
                dispatched_buffer,
                gate_projs[local_expert],
                up_projs[local_expert],
                down_projs[local_expert],
                expert_token_counts,
                global_expert_idx_table,
                expert_region_offsets,
                local_expert,
                /*use_region_offsets=*/true,
                compute_kernel_config,
                /*output=*/expert_outputs);
        }
        return expert_outputs;
    }

    auto expert_outputs = dispatched_buffer;
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        auto tokens = ttnn::extract(
            dispatched_buffer,
            expert_region_offsets,
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            max_dispatched_tokens_per_expert);
        auto ffn_out =
            ffn(tokens,
                gate_projs[local_expert],
                up_projs[local_expert],
                down_projs[local_expert],
                expert_token_counts,
                global_expert_idx_table,
                expert_region_offsets,
                local_expert,
                /*use_region_offsets=*/false,
                compute_kernel_config,
                std::nullopt);
        expert_outputs = ttnn::insert(
            expert_outputs, ffn_out, expert_region_offsets, expert_token_counts, global_expert_idx_table, local_expert);
    }
    return expert_outputs;
}

}  // namespace

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
    return unified_routed_expert_moe_impl(
        dispatched_buffer,
        expert_region_offsets,
        expert_token_counts,
        global_expert_idx_table,
        gate_projs,
        up_projs,
        down_projs,
        max_dispatched_tokens_per_expert,
        compute_kernel_config,
        &unified_routed_expert_ffn);
}

ttnn::Tensor unified_routed_expert_moe_wh(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return unified_routed_expert_moe_impl(
        dispatched_buffer,
        expert_region_offsets,
        expert_token_counts,
        global_expert_idx_table,
        gate_projs,
        up_projs,
        down_projs,
        max_dispatched_tokens_per_expert,
        compute_kernel_config,
        &unified_routed_expert_ffn_wh);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
