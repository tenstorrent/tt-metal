// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/data_movement/repeat_codegen/device/repeat_codegen_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct RepeatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RepeatCodegenParams& operation_attributes,
        const RepeatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
