// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "groupnorm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::normalization::group_norm {

struct GroupNormShardedSharedVariables {
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    tt::tt_metal::CBHandle cb_in0{};
    tt::tt_metal::CBHandle cb_output{};
    uint32_t num_cores = 0;
    CoreCoord grid_size;
};

struct GroupNormShardedProgramFactory {
    using shared_variables_t = GroupNormShardedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GroupNormParams& operation_attributes, const GroupNormInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GroupNormParams& operation_attributes,
        const GroupNormInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::normalization::group_norm
