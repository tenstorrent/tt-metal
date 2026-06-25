// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TODO(nuked-op matmul): minimal stub
// The matmul op was nuked, but the MatmulProgramConfig variant (and its members) is a
// shared host type still consumed by surviving code: the global_circular_buffer matmul-1d
// GCB API + impl, conv2d / conv_transpose2d, and the deepseek routed_expert_ffn factories.
// This reproduces the config structs verbatim (they only depend on surviving headers) so
// those consumers compile until matmul is recreated from scratch.

#pragma once

#include "tt-metalium/core_coord.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::matmul {

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
};

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
        },
        config);
}

}  // namespace ttnn::operations::matmul
