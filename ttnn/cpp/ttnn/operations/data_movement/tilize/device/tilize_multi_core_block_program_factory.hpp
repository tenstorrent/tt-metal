// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "tilize_device_operation_types.hpp"

namespace ttnn::operations::data_movement::program {

struct TilizeMultiCoreBlockProgramFactory {
    using shared_variables_t = tilize::program::MultiCoreSharedVariables::shared_variables_t;
    using cached_program_t =
        ttnn::device_operation::CachedProgram<TilizeMultiCoreBlockProgramFactory::shared_variables_t>;
    using operation_attributes_t = tilize::operation_attributes_t;

    static cached_program_t create(
        const tilize::operation_attributes_t& operation_attributes,
        const tilize::tensor_args_t& tensor_args,
        const tilize::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const tilize::operation_attributes_t& operation_attributes,
        const tilize::tensor_args_t& tensor_args,
        const tilize::tensor_return_value_t& tensor_return_value);
};
}  // namespace ttnn::operations::data_movement::program
