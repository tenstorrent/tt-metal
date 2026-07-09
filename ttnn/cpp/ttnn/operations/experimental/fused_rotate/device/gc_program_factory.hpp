// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gc_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct FusedGcSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct FusedGcProgramFactory {
    using shared_variables_t = FusedGcSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const FusedGcParams& attrs, const FusedGcInputs& inputs, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const FusedGcParams& attrs, const FusedGcInputs& inputs, Tensor& output);
};

}  // namespace ttnn::experimental::prim
