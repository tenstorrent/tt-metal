// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>

#include "generic_op_device_operation.hpp"
#include "tt-metalium/kernel_types.hpp"

namespace ttnn::operations::generic {
GenericOpDeviceOperation::GenericProgram::cached_program_t GenericOpDeviceOperation::GenericProgram::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{operation_attributes};
    return {std::move(program), {}};
}

void GenericOpDeviceOperation::GenericProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::generic
