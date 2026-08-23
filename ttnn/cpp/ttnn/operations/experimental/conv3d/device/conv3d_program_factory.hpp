// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "conv3d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct Conv3dProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
