// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct NonZeroIndicesProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NonzeroParams& operation_attributes, const NonzeroInputs& tensor_args, NonzeroResult& output_tensors);
};

}  // namespace ttnn::prim
