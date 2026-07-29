// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"

#include "scatter_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ScatterProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ScatterParams& args, const ScatterInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
