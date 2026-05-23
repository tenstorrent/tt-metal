// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"

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
    // Pick the largest chunk_M_tiles that:
    //   (a) is one of {16, 32, 64} — per_core_M must be >= 2 (subblock_h = 1
    //       so per_core_M = 1 hits an edge case in matmul_phase_v3's L1_ACC
    //       accumulation that drops PCC to ~0). Smallest viable per_core_M
    //       is therefore 2, chunk_M_tiles = 2 * GRID_Y = 16;
    //   (b) divides x's M_tiles_full evenly so every chunk does real work;
    //   (c) is no larger than 64 tiles (L1 cap on cb_partials_d).
    //
    // For DS-V3 dims this picks 64 for {2k, 4k, 8k, 16k}, 32 for {1k, 5k,
    // 25k}, 16 for {3.2k after 16-tile pad}. The wrapper in
    // tt_routed_expert.py pads up to the next 16-tile (= 512-row) boundary
    // when M is not already aligned.
    constexpr uint32_t kGridY = 8;
    constexpr uint32_t kMinChunkMTiles = 16;  // per_core_M >= 2
    constexpr uint32_t kMaxChunkMTiles = 64;  // per_core_M <= 8 (L1 cap)
    const uint32_t M_tiles_full = x.padded_shape()[-2] / 32;
    uint32_t chunk_M_tiles = kMinChunkMTiles;
    for (uint32_t cand = kMaxChunkMTiles; cand >= kMinChunkMTiles; cand -= kGridY) {
        if (M_tiles_full % cand == 0) {
            chunk_M_tiles = cand;
            break;
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
