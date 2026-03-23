// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "offset_cumsum_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct OffsetCumsumSharedVariables {
    tt::tt_metal::KernelHandle kernel_id = 0;
    CoreCoord core;
};

struct OffsetCumsumProgramFactory {
    using shared_variables_t = OffsetCumsumSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    static cached_program_t create(
        const OffsetCumsumParams& operation_attributes,
        const Tensor& input,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const OffsetCumsumParams& operation_attributes,
        const Tensor& input,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
