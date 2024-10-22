// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_coord.hpp"

namespace ttnn::operations::normalization {

struct SoftmaxDefaultProgramConfig{
};
struct SoftmaxShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w;
    std::size_t block_h;
    std::size_t block_w;
};

using SoftmaxProgramConfig = std::variant<
    SoftmaxDefaultProgramConfig,
    SoftmaxShardedMultiCoreProgramConfig
>;

}  // namespace ttnn::operations::normalization
