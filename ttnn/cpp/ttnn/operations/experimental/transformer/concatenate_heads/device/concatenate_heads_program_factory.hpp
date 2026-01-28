// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concatenate_heads_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct ConcatenateHeadsSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    uint32_t num_cores_r = 0;
    uint32_t num_cores_c = 0;
};

struct ConcatenateHeadsProgramFactory {
    using shared_variables_t = ConcatenateHeadsSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ConcatenateHeadsParams& operation_attributes, const ConcatenateHeadsInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ConcatenateHeadsParams& operation_attributes,
        const ConcatenateHeadsInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
