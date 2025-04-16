// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/core_coord.hpp>

using namespace tt::tt_metal;
namespace ttnn::operations::normalization {

struct GroupNormMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    MathFidelity math_fidelity;
    DataType im_data_format;
    DataType out_data_format;
    bool inplace;
    Layout output_layout;
    int num_out_blocks;
};
struct GroupNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    MathFidelity math_fidelity;
    DataType im_data_format;
    DataType out_data_format;
    bool inplace;
    Layout output_layout;
};

using GroupNormProgramConfig = std::variant<GroupNormMultiCoreProgramConfig, GroupNormShardedMultiCoreProgramConfig>;

}  // namespace ttnn::operations::normalization
