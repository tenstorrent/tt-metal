// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "extract_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract {

struct ExtractSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct ExtractProgramFactory {
    using shared_variables_t = ExtractSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ExtractParams& operation_attributes, const ExtractInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ExtractParams& operation_attributes,
        const ExtractInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract
