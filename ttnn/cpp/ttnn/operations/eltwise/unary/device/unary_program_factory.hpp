// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unary_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct UnaryProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output);
};

struct UnarySubCoreGridProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
