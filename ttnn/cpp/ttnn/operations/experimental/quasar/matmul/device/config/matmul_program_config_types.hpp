// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/buffer_types.hpp"
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
    std::optional<tt::tt_metal::CoreRangeSet> allowed_worker_cores = std::nullopt;
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
    std::optional<tt::tt_metal::CoreRangeSet> allowed_worker_cores = std::nullopt;
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
    tt::tt_metal::CoreRangeSet hop_cores;
    std::size_t num_global_cb_receivers{};
    bool untilize_out{};
    std::optional<tt::tt_metal::CoreRangeSet> allowed_worker_cores = std::nullopt;
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
    std::optional<tt::tt_metal::CoreRangeSet> allowed_worker_cores = std::nullopt;
};

// Placement-first config for the Quasar-native matmul (GH#41910): the caller names the clusters and
// the C slice (in 32x32 tiles) each produces in one go; the factory assigns one batch's C slices to
// `cores` as contiguous runs, or, when batches with their own B have too few C slices each, to groups of the cores
// that each take a run of the batches. Edge C slices are clipped on read/write, so any M / N works. Within a
// cluster the C slice's subblocks are assigned round-robin to the compute threads (NEOs). A bias row and the
// activation are applied in DST before the final pack.
// Batches of A that use one B are stacked into one [batch * M x N] GEMM.
// A sharded C takes any placement (the writer reaches its shards through the tensor accessor); auto prefers placements
// whose C slices are its shards. Limits: no untilize, 32x32 tiles only.
struct MatmulUnifiedProgramConfig {
    // Empty = auto placement: the cores, C slice and orientation are chosen by a cost estimate over the natural
    // placements (the other fields are kept).
    tt::tt_metal::CoreRangeSet cores;
    // C slice (in tiles) each core produces in one go. Unset = auto: of a sharded C's shard shape and the C slices
    // whose dims divide M / N split over the bounding box of `cores`, the one whose plan has the lowest estimated
    // cycles (auto fields are chosen together, see plan_unified_matmul).
    std::optional<std::size_t> C_slice_M_tiles;
    std::optional<std::size_t> C_slice_N_tiles;
    // K tiles multiplied per accumulation step: one A slice and one B slice are resident in L1 at a time and
    // the partial sums round-trip L1 between steps. Must divide K_tiles. Unset = auto (lowest estimated cycles).
    std::optional<std::size_t> K_chunk_tiles;
    // Subblock: the C slice's tiles accumulated in DST at once; holds <= 8 tiles (4 with fp32
    // accumulation). Need not divide the C slice: it is padded up to subblock multiples and the
    // overshoot is clipped on write. Set both or neither; unset = auto (lowest estimated cycles).
    std::optional<std::size_t> subblock_M_tiles;
    std::optional<std::size_t> subblock_N_tiles;
    // Order `cores` are walked when handing out C slices: ROW_MAJOR x fastest, COL_MAJOR y fastest. A
    // sharded C allocated without a shard spec gets this shard orientation.
    tt::tt_metal::ShardOrientation orientation = tt::tt_metal::ShardOrientation::ROW_MAJOR;
    // Compute threads per core (Quasar NEOs running the compute kernel); 1, 2 or 4. Unset = auto: 4 on Quasar,
    // 1 elsewhere (Wormhole / Blackhole have one compute engine per core).
    std::optional<std::size_t> num_compute_threads;
    // Groups the cores split into when batches have their own B: each group takes a run of the batches and splits every
    // one of them over its cores like a single batch (1 = every core takes every batch). At most the batches and the
    // cores. Unset = auto (lowest estimated cycles; more than one only when a batch has too few C slices for the
    // cores).
    std::optional<std::size_t> num_batch_groups;
    // Applied to the finished sum in DST, after the bias, before the final pack.
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation = std::nullopt;
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
        return tt::tt_metal::CoreRangeSet(
            tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1)));
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
