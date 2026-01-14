// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "conv3d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::conv3d::program {

struct Conv3dSharedVariables {
    uint32_t num_cores = 0;
    std::vector<CoreCoord> cores;
    CoreCoord grid_size;
    tt::tt_metal::KernelHandle reader_kernels_id = 0;
    tt::tt_metal::KernelHandle writer_kernels_id = 0;
    tt::tt_metal::KernelHandle compute_kernels_id = 0;
};

struct Conv3dProgramFactory {
    using shared_variables_t = Conv3dSharedVariables;
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

}  // namespace ttnn::operations::experimental::conv3d::program
