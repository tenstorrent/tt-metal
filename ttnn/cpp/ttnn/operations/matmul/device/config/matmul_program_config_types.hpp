// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/core_coord.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::matmul {

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1
// for in1 iff B=1 for in0 (ie. single core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool transpose_mcast{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
    bool fuse_batch = true;
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
    // When true, factory emits TILE_PACK_ROW_MAJOR to compute + writer kernels so the pack
    // LLK writes tiles at absolute CB offsets row-first across all N-subblocks, and the
    // writer reads per-M-row-group. Unlocks multi-row subblocks (out_subblock_h > 1 with
    // out_subblock_w < per_core_N) by decoupling subblock shape from writer tile order.
    // Name disambiguates from Layout::ROW_MAJOR (untilized element layout): this flag is
    // about the ORDER tiles are packed within the output block, not the tensor layout.
    bool tile_pack_row_major = false;
};

// 1D mcast matmul program config.
//
// When `gather_in0 == false`, `compute_with_storage_grid_size` describes the size of the
// rectangular grid of worker cores that the multicast paths will use, anchored at (0, 0) on
// the device, or at the bounding-box start of the active sub-device when `sub_device_id` is
// set on the op. The 1D mcast factory targets a single bounding-box rectangle for multicast
// and the per-core index math assumes a single contiguous row-major rectangle, so when
// `sub_device_id` is provided the sub-device's worker cores must themselves form a single
// rectangle. Non-rectangular sub-device grids are rejected at validate time.
//
// When `gather_in0 == true`, `compute_with_storage_grid_size` is ignored and the gather path
// can run on any sub-device worker layout, including non-rectangular ones.
struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t out_block_h{};
    std::size_t out_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool fuse_batch{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
    bool mcast_in0{};
    bool gather_in0{};
    CoreRangeSet hop_cores;
    std::size_t num_global_cb_receivers{};
    bool untilize_out{};
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
    // See MatmulMultiCoreReuseMultiCastProgramConfig::tile_pack_row_major.
    bool tile_pack_row_major = false;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig {
    std::size_t in0_block_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>;

// Ensures allowed_worker_cores is populated on every config variant that supports it.
// If allowed_worker_cores is already set, it is left unchanged.  Otherwise it is
// synthesized from compute_with_storage_grid_size (or from the device grid for
// MatmulMultiCoreProgramConfig).  After this call, factories can read
// config.allowed_worker_cores.value() unconditionally.
inline void normalize_program_config(MatmulProgramConfig& config, const CoreCoord& device_grid) {
    auto make_crs = [](const CoreCoord& grid) {
        return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    };
    std::visit(
        [&](auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (
                std::is_same_v<T, MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<T, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<T, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (!c.allowed_worker_cores.has_value()) {
                    c.allowed_worker_cores = make_crs(c.compute_with_storage_grid_size);
                }
            } else if constexpr (std::is_same_v<T, MatmulMultiCoreProgramConfig>) {
                if (!c.allowed_worker_cores.has_value()) {
                    c.allowed_worker_cores = make_crs(device_grid);
                }
            }
            // DRAM-sharded configs have no grid fields to normalize.
        },
        config);
}

}  // namespace ttnn::operations::matmul
