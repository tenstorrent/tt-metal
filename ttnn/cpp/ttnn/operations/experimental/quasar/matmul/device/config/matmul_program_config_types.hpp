// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/core_coord.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::experimental::quasar::matmul {

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1
// for in1 iff B=1 for in0 (ie. single core)
struct MatmulMultiCoreReuseProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    std::optional<CoreRangeSet> allowed_worker_cores = std::nullopt;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
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
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
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

// Placement-first program config for the Quasar-native matmul (stage A of GH#41910).
//
// GEMM vocabulary, all sizes in 32x32 tiles: C[M x N] = A[M x K] x B[K x N]. The caller describes the
// work directly instead of picking a 1D / 2D / DRAM-sharded strategy:
//   - `cores`                    the clusters that take part;
//   - `MN_chunk_M_tiles` / `MN_chunk_N_tiles` the MN chunk of C (in tiles) each cluster produces in one go.
// The factory walks the MN chunks of one batch (across N, then down M) and hands that walk to
// `cores` in enumeration order (x fastest when `row_major_cores`, y fastest otherwise) as contiguous
// runs; when there are fewer MN chunks than cores the trailing cores idle, when there are more each core
// produces several. Blocks on the right / bottom edge are computed at full size and clipped on read
// and write, so any M / N works. Every operand is addressed by tile index through the tensor accessor, so interleaved,
// L1-sharded and DRAM-sharded tensors all take the same kernels. The legacy strategies are particular
// choices of (cores, MN_chunk_M_tiles, MN_chunk_N_tiles): e.g. a 1D "mcast_in0" matmul is MN_chunk_M_tiles = M_tiles on
// a row of cores, a 2D matmul is a rectangle of cores with MN_chunk_M_tiles x MN_chunk_N_tiles MN chunks.
//
// Stage A limits: one NEO, one reader and one writer per cluster; no data sharing between clusters;
// no bias (the op applies it as a separate add), no fused activation, no untilize, 32x32 tiles only,
// sharded output needs batch 1 and exactly one MN chunk per core.
struct MatmulUnifiedProgramConfig {
    CoreRangeSet cores;
    std::size_t MN_chunk_M_tiles{};
    std::size_t MN_chunk_N_tiles{};
    // K tiles accumulated per K chunk (one A slice + one B slice in L1 at a time); must divide K_tiles.
    // 0 = auto: the largest divisor of K_tiles <= 8 whose rings fit L1.
    std::size_t K_chunk_tiles = 0;
    // Subblock: the MN chunk's tiles accumulated in DST at once; must divide MN_chunk_M_tiles / MN_chunk_N_tiles and
    // hold
    // <= 8 tiles (4 with fp32 accumulation). 0 for both = auto.
    std::size_t subblock_M_tiles = 0;
    std::size_t subblock_N_tiles = 0;
    bool row_major_cores = true;
};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig,
    MatmulUnifiedProgramConfig>;

// Ensures allowed_worker_cores is populated on every config variant that supports it.
// If allowed_worker_cores is already set, it is left unchanged.  Otherwise it is
// synthesized from compute_with_storage_grid_size (or from the device grid for
// MatmulMultiCoreProgramConfig).  After this call, factories can read
// config.allowed_worker_cores.value() unconditionally.
inline void normalize_program_config(MatmulProgramConfig& config, const tt::tt_metal::CoreCoord& device_grid) {
    auto make_crs = [](const tt::tt_metal::CoreCoord& grid) {
        return CoreRangeSet(CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1)));
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
            // DRAM-sharded and unified configs have no grid fields to normalize.
        },
        config);
}

}  // namespace ttnn::operations::experimental::quasar::matmul
