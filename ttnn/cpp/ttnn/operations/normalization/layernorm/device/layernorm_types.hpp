// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::normalization {

enum class LayerNormType { LAYERNORM, RMSNORM };

enum class DistributedLayerNormStage { NOT_DISTRIBUTED, PRE_ALL_GATHER, POST_ALL_GATHER };

struct LayerNormDefaultProgramConfig {};
struct LayerNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w;
    std::size_t block_h;
    std::size_t block_w;
    bool inplace;
};

using LayerNormProgramConfig = std::variant<LayerNormDefaultProgramConfig, LayerNormShardedMultiCoreProgramConfig>;

}  // namespace ttnn::operations::normalization
