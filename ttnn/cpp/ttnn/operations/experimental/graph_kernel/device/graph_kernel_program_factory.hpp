// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "graph_kernel_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct GraphKernelProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GraphKernelParams& operation_attributes, const GraphKernelInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::experimental::prim
