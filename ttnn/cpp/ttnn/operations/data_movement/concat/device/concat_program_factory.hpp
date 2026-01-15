// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::data_movement::concat::program {

// Shared variables for interleaved concat
struct ConcatSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct ConcatProgramFactory {
    using shared_variables_t = ConcatSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ConcatParams& operation_attributes,
        const ConcatInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::concat::program
