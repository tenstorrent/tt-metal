// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_coord.h"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::matmul {

// TODO: Uplift this to support fused activation and bias
// TODO: Uplift this to support bcast batch for in1; currently, only allows B=1 for in1 iff B=1 for in0 (ie. single
// core)
struct MatmulMultiCoreReuseProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
};

struct MatmulMultiCoreReuseMultiCastProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool transpose_mcast;
    std::optional<unary::UnaryWithParam> fused_activation;
    bool fuse_batch = true;
};

struct MatmulMultiCoreReuseMultiCast1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w;
    std::size_t out_subblock_h;
    std::size_t out_subblock_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    bool fuse_batch;
    std::optional<unary::UnaryWithParam> fused_activation;
    bool mcast_in0;
};

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig {
    std::size_t in0_block_w;
    std::size_t per_core_M;
    std::size_t per_core_N;
    std::optional<unary::UnaryWithParam> fused_activation;
};

struct MatmulMultiCoreProgramConfig {};

struct MatmulMultiCoreNonOptimizedReuseProgramConfig {};

using MatmulProgramConfig = std::variant<
    MatmulMultiCoreProgramConfig,
    MatmulMultiCoreNonOptimizedReuseProgramConfig,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>;

}  // namespace ttnn::operations::normalization
