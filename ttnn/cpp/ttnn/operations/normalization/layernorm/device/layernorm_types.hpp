// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

enum class LayerNormType { LAYERNORM, RMSNORM };

enum class DistributedLayerNormStage { NOT_DISTRIBUTED, PRE_ALL_GATHER, POST_ALL_GATHER };

struct LayerNormDefaultProgramConfig {
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
    // > 1 (RMSNorm on an interleaved TILE input only): split every tile row across width_split cores, which
    // exchange their partial mean of squares (LayerNormWidthSplitProgramFactory). 1 = one core per tile row.
    std::size_t width_split = 1;
};
struct LayerNormShardedMultiCoreProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w{};
    std::size_t block_h{};
    std::size_t block_w{};
    bool inplace{};
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
};

using LayerNormProgramConfig = std::variant<LayerNormDefaultProgramConfig, LayerNormShardedMultiCoreProgramConfig>;

}  // namespace ttnn::prim
