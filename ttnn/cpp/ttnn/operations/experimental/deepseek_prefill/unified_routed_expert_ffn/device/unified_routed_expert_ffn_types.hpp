// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

// Attributes (the constants known at host time).
struct UnifiedRoutedExpertFfnParams {
    // The compute kernel chunks the M axis into pieces of this many tiles so a
    // single matmul fits in per-core L1. 64 (= 2048 tokens) is the maximum that
    // keeps DeepSeek V3 routed-expert dims inside Blackhole L1.
    uint32_t chunk_M_tiles = 64;

    // Local expert id used to index `global_expert_idx_table` at runtime
    // (kernel reads global_id = idx_table[local_expert_id], then count =
    // counts[global_id]).
    uint32_t local_expert_id = 0;

    // When true (fused extract+FFN+insert path): kernels add
    // start_tile_row = expert_region_offsets[global_id]/32 to all DRAM tile
    // indices so x reads and output writes hit this expert's slice of a
    // shared dispatched_buffer. CB_IDX_SCRATCH page holds both idx and
    // region_offsets.
    //
    // When false (unfused path): x is the already-extracted per-expert
    // tokens tensor and output is a fresh per-expert tensor, both starting
    // at row 0. Kernels use start_tile_row=0 and skip the region_offsets
    // DRAM read. CB_IDX_SCRATCH holds only idx — saves the region_offsets
    // page slot, freeing L1 for the 256-expert / 32-per-chip configuration
    // where the combined-page CB pushed past the L1 budget and clashed
    // with allocated L1 buffers at program create.
    bool use_region_offsets = true;

    // Compute grid shape. Defaults are the Blackhole layout (11x8 = 88 cores).
    // The Wormhole variant uses an 8x8 = 64-core layout because WH worker
    // grid is 8x8 (compute_with_storage_grid_size).
    uint32_t grid_x = 11;
    uint32_t grid_y = 8;
    // K-dimension inner block tile count for the gate/up matmuls. Must divide
    // K_gate_tiles (= emb / TILE). The Blackhole layout uses 16; the WH path
    // selects the largest divisor of K_gate_tiles that fits L1 (e.g. 10 for
    // emb=2880 -> K_gate_tiles=90).
    uint32_t in0_block_w_gu = 16;

    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "chunk_M_tiles", "local_expert_id", "use_region_offsets", "grid_x", "grid_y", "in0_block_w_gu");
    auto attribute_values() const {
        return std::forward_as_tuple(
            chunk_M_tiles, local_expert_id, use_region_offsets, grid_x, grid_y, in0_block_w_gu);
    }
};

// Tensors fed into the op.
//
// x is the (M_max, K=emb) dispatched-token buffer for this expert. Only the
// first `counts[global_expert_idx_table[local_expert_id]]` rows are valid;
// the rest is padding the FFN kernels must skip.
//
// gate_proj/up_proj/down_proj are the (K=emb, N=hidden), (K=emb, N=hidden),
// and (K=hidden, N=emb) weight tensors.
//
// counts/global_expert_idx_table are the device-side count buffers; the
// kernel reads them at runtime to skip unused chunks.
struct UnifiedRoutedExpertFfnInputs {
    Tensor x;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    Tensor counts;
    Tensor global_expert_idx_table;
    // Per-global-expert starting M-row in x (in tokens). The reader reads
    // region_offsets[global_expert_id] device-side and adds it to the M-tile
    // index of each DRAM read, so the FFN can operate on a slice of a
    // larger shared buffer (e.g. the full dispatched_buffer) without a
    // preceding extract op. Same for the writer's output tile_idx — output
    // writes land at the same offset, so a separate insert op is also
    // unnecessary. For single-expert tests where x is already the expert's
    // tokens, pass a single-element [0] offsets tensor.
    Tensor expert_region_offsets;
    std::optional<Tensor> optional_output;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
