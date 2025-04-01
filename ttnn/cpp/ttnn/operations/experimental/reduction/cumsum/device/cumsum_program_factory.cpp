// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumsum_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::reduction {

CumSumDeviceOperation::SingleCore::cached_program_t CumSumDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program;

    // Scaffold: No 'real' program for now
    return {std::move(program), {}};
}

void CumSumDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::experimental::reduction
