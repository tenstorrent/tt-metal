// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeMultiCoreDefaultProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
