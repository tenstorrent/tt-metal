// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim {
struct QuasarMatmulSharedVariables {};

struct QuasarMatmulProgramFactory {
    static cached_program_t create(
        const MatmulParams& operation_attributes, const MatmulInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MatmulParams& operation_attributes,
        const MatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
