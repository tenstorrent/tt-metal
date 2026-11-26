// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "reshape_device_operation_types.hpp"

namespace ttnn::prim {

struct ReshapeTileProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::ReshapeOnDeviceParams& operation_attributes,
        const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
        ttnn::Tensor& output_tensor);
};

}  // namespace ttnn::prim
