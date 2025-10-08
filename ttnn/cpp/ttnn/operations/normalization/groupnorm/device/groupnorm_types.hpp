// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/functions.hpp"

namespace ttnn::operations::normalization {

struct GroupNormMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    DataType im_data_format{DataType::INVALID};
    DataType out_data_format{DataType::INVALID};
    bool inplace{};
    Layout output_layout{Layout::INVALID};
    int num_out_blocks{};
};
struct GroupNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    DataType im_data_format{DataType::INVALID};
    DataType out_data_format{DataType::INVALID};
    bool inplace{};
    Layout output_layout{Layout::INVALID};
};

using GroupNormProgramConfig = std::variant<GroupNormMultiCoreProgramConfig, GroupNormShardedMultiCoreProgramConfig>;

}  // namespace ttnn::operations::normalization
