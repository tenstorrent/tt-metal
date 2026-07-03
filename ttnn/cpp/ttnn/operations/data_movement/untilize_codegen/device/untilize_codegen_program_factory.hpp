// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/data_movement/untilize_codegen/device/untilize_codegen_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct UntilizeCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeCodegenParams& operation_attributes,
        const UntilizeCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
