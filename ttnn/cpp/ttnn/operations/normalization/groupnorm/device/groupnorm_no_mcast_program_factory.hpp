// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "groupnorm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::normalization::group_norm {

struct GroupNormNoMcastSharedVariables {
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> reader_sender_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> reader_receiver_kernel_ids;
    std::vector<CoreCoord> core_coords;
    CoreCoord grid_size;
    std::vector<std::vector<CoreCoord>> mcast_groups;
    uint32_t groupnorm_mode = 0;
    tt::tt_metal::CBHandle cb_reciprocals_handle{};
};

struct GroupNormNoMcastProgramFactory {
    using shared_variables_t = GroupNormNoMcastSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::normalization::group_norm
