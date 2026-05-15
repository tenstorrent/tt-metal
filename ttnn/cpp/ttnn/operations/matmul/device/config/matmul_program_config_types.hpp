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

struct MatmulMultiCoreProgramConfig {};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>;

}  // namespace ttnn::operations::matmul
