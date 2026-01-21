// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "variance_w_rm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::reduction::variance_w_rm::program {

struct VarianceWRmSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    CoreRangeSet all_cores;
    uint32_t num_cores = 0;
};

struct VarianceWRmProgramFactory {
    using shared_variables_t = VarianceWRmSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const VarianceWRmParams& operation_attributes,
        const VarianceWRmInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const VarianceWRmParams& operation_attributes,
        const VarianceWRmInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::reduction::variance_w_rm::program
