// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_post_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::normalization::layernorm_post_all_gather::program {

struct LayerNormPostAllGatherSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t num_cores{};
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_tile_rows_per_core_group_1{};
    uint32_t num_tile_rows_per_core_group_2{};
    uint32_t Wt{};
    uint32_t stats_tiles_cols{};
};

struct LayerNormPostAllGatherProgramFactory {
    using shared_variables_t = LayerNormPostAllGatherSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::normalization::layernorm_post_all_gather::program
