// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_combine_program_factory.hpp"
#include "prefill_combine_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_combine {

PrefillCombineDeviceOperation::PrefillCombineProgramFactory::cached_program_t
PrefillCombineDeviceOperation::PrefillCombineProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // TODO: Implement program creation
    // - Create program
    // - Add kernels (reader, writer)
    // - Configure CBs
    // - Set runtime args
    // - Return cached program with shared variables

    TT_THROW("PrefillCombineProgramFactory::create not yet implemented");
}

void PrefillCombineDeviceOperation::PrefillCombineProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // TODO: Implement runtime argument override
    // - Update kernel runtime args for new tensor addresses
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine
