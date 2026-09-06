// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gelu_bw_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::operations::unary_backward::gelu_bw {

struct GeluBwProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GeluBwParams& args, const GeluBwInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::operations::unary_backward::gelu_bw
