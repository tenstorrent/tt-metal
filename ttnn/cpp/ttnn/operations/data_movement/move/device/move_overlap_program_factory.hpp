// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "move_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

// Program factory for MULTI_CORE_OVERLAP strategy
struct MoveOverlapProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id = 0;
        uint32_t num_cores = 0;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MoveOperationAttributes& operation_attributes,
        const MoveTensorArgs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MoveOperationAttributes& operation_attributes,
        const MoveTensorArgs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
