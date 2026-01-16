// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layernorm_fused_rm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::normalization::layernorm_fused_rm::program {

struct LayernormFusedRmSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    CoreRangeSet all_cores;
    uint32_t num_cores = 0;
};

struct LayernormFusedRmProgramFactory {
    using shared_variables_t = LayernormFusedRmSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const LayernormFusedRmParams& operation_attributes,
        const LayernormFusedRmInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const LayernormFusedRmParams& operation_attributes,
        const LayernormFusedRmInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::normalization::layernorm_fused_rm::program
