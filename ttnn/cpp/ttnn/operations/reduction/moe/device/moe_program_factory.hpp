// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/reduction/moe/device/moe_device_operation_types.hpp"

namespace ttnn::prim {

struct MoeProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MoeParams& operation_attributes, const MoeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
