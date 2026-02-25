// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "copy_device_operation_types.hpp"

namespace ttnn::prim {

struct CopySharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle unary_writer_kernel_id{};
    std::vector<CoreCoord> cores;
};

struct CopyProgramFactory {
    using shared_variables_t = CopySharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const CopyParams& operation_attributes, const CopyInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const CopyParams& operation_attributes,
        const CopyInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
