// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumsum_device_operation.hpp"

namespace ttnn::operations::experimental::reduction {

CumSumDeviceOperation::SingleCore::cached_program_t CumSumDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program;

    // TODO: Device set up

    return program;
}

}  // namespace ttnn::operations::experimental::reduction
