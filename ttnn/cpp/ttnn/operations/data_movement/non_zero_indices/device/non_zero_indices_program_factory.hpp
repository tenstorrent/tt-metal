// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation_types.hpp"

namespace ttnn::prim {

struct NonZeroIndicesSharedVariables {
    tt::tt_metal::KernelHandle kernel_id{};
    tt::tt_metal::CoreCoord core;
    uint32_t page_size{};
};

struct NonZeroIndicesProgramFactory {
    using shared_variables_t = NonZeroIndicesSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const NonzeroParams& operation_attributes, const NonzeroInputs& tensor_args, NonzeroResult& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const NonzeroParams& operation_attributes,
        const NonzeroInputs& tensor_args,
        NonzeroResult& output_tensors);
};

}  // namespace ttnn::prim
