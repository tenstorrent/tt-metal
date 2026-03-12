// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "masked_bincount_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MaskedBincountSharedVariables {
    tt::tt_metal::KernelHandle kernel_id_brisc = 0;
    tt::tt_metal::KernelHandle kernel_id_ncrisc = 0;
    std::vector<CoreCoord> all_cores_vec;
    CoreCoord collector_core;
    uint32_t num_cores;
};

struct MaskedBincountProgramFactory {
    using shared_variables_t = MaskedBincountSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MaskedBincountParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MaskedBincountParams& operation_attributes,
        const Tensor& input,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
