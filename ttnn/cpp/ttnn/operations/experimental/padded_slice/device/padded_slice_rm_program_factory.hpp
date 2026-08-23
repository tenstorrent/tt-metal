// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "padded_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct PaddedSliceRMProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::experimental::prim
