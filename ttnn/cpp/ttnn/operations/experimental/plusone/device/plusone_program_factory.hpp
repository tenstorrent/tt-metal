// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "plusone_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct PlusOneSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct PlusOneProgramFactory {
    using shared_variables_t = PlusOneSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const PlusoneParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const PlusoneParams& operation_attributes,
        const Tensor& input,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
