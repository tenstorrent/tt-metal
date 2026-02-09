// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/split/device/split_device_operation_types.hpp"

namespace ttnn::operations::data_movement::split::program {

struct SplitSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    uint32_t num_cores_r;
    uint32_t num_cores_c;
    uint32_t start_core_x;
    uint32_t start_core_y;
};

struct SplitProgramFactory {
    using shared_variables_t = SplitSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const split::SplitParams& operation_attributes,
        const split::SplitInputs& tensor_args,
        split::tensor_return_value_t& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const split::SplitParams& operation_attributes,
        const split::SplitInputs& tensor_args,
        split::tensor_return_value_t& output_tensors);
};

}  // namespace ttnn::operations::data_movement::split::program
