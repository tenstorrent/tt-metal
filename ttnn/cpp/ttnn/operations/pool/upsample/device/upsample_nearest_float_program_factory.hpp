// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "upsample_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct UpsampleNearestFloatSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    CoreRangeSet all_cores;
    uint32_t num_cores = 0;
};

struct UpsampleNearestFloatProgramFactory {
    using shared_variables_t = UpsampleNearestFloatSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const UpsampleParams& operation_attributes, const Tensor& input, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UpsampleParams& operation_attributes,
        const Tensor& input,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
