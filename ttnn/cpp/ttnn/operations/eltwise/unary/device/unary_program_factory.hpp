// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unary_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::unary::program {

struct UnaryProgramFactory {
    static tt::tt_metal::ProgramDescriptor create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::unary::program
