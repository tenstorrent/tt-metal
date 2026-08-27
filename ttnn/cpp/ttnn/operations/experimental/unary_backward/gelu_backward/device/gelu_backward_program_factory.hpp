// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gelu_backward_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct GeluBackwardProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GeluBackwardParams& args, const GeluBackwardInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::experimental::prim
