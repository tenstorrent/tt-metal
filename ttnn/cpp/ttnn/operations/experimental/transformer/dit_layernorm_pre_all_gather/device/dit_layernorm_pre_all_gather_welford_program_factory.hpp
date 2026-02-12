// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dit_layernorm_pre_all_gather_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct PreAllGatherWelfordSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    uint32_t num_cores = 0;
    CoreCoord grid_size;
};

struct PreAllGatherWelfordProgramFactory {
    using shared_variables_t = PreAllGatherWelfordSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const DitLayernormPreAllGatherParams& operation_attributes, const Tensor& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const DitLayernormPreAllGatherParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
