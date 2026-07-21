// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/data_movement/repeat/device/repeat_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct RepeatProgramFactoryLastDim {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RepeatParams& operation_attributes, const RepeatInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
