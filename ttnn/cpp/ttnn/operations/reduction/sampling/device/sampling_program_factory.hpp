// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/sampling/device/sampling_device_operation_types.hpp"

namespace ttnn::prim {

struct SamplingSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    std::vector<tt::tt_metal::CoreCoord> cores;
};

struct SamplingProgramFactory {
    using shared_variables_t = SamplingSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const SamplingParams& operation_attributes, const SamplingInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const SamplingParams& operation_attributes,
        const SamplingInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
