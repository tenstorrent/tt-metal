// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fused_rmsnorm_pre_all_gather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct FusedRMSNormPreAllGatherSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct FusedRMSNormPreAllGatherProgramFactory {
    using shared_variables_t = FusedRMSNormPreAllGatherSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FusedRmsnormPreAllGatherParams& operation_attributes,
        const FusedRmsnormPreAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FusedRmsnormPreAllGatherParams& operation_attributes,
        const FusedRmsnormPreAllGatherInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
