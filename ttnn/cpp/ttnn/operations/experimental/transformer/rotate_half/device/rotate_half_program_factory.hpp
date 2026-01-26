// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rotate_half_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RotateHalfSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    CoreCoord core;
};

struct RotateHalfProgramFactory {
    using shared_variables_t = RotateHalfSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RotateHalfParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RotateHalfParams& operation_attributes,
        const Tensor& input,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
