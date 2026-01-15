// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "tilize_device_operation_types.hpp"

namespace ttnn::prim {

struct TilizeMultiCoreInterleavedProgramFactory {
    using shared_variables_t = ttnn::prim::MultiCoreSharedVariables::shared_variables_t;
    using cached_program_t =
        ttnn::device_operation::CachedProgram<ttnn::prim::MultiCoreSharedVariables::shared_variables_t>;

    static cached_program_t create(
        const ttnn::prim::operation_attributes_t& operation_attributes,
        const ttnn::prim::tensor_args_t& tensor_args,
        const ttnn::prim::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::prim::operation_attributes_t& operation_attributes,
        const ttnn::prim::tensor_args_t& tensor_args,
        const ttnn::prim::tensor_return_value_t& tensor_return_value);
};
}  // namespace ttnn::prim
