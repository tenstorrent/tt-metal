// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "padded_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct PaddedSliceTileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::experimental::prim
