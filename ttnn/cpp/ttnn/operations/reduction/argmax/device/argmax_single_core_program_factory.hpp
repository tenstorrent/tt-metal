// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "argmax_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct ArgMaxSingleCoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    std::vector<CoreCoord> cores;
};

struct ArgMaxSingleCoreProgramFactory {
    using shared_variables_t = ArgMaxSingleCoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ArgmaxParams& operation_attributes,
        const ArgmaxInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
