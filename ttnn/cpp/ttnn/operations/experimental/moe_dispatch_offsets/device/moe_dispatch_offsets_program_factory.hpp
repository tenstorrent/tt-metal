// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_dispatch_offsets_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MoeDispatchOffsetsSharedVariables {
    tt::tt_metal::KernelHandle kernel_id = 0;
    CoreCoord core;
};

struct MoeDispatchOffsetsProgramFactory {
    using shared_variables_t = MoeDispatchOffsetsSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MoeDispatchOffsetsParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MoeDispatchOffsetsParams& operation_attributes,
        const Tensor& input,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
