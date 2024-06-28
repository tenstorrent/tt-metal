// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"

namespace ttnn::operations::examples {
ExampleDeviceOperation::SingleCore::cached_program_t ExampleDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    tt_metal::Program program = tt_metal::CreateProgram();

    return {std::move(program), {.some_variable_from_create_to_use_in_override_runtime_arguments = 1}};
}

void ExampleDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& some_variable_from_create_to_use_in_override_runtime_arguments =
        cached_program.shared_variables.some_variable_from_create_to_use_in_override_runtime_arguments;
}

}  // namespace ttnn::operations::examples
