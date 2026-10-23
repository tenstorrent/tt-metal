// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim::qsr {

// Metal 2.0 (MetalV2FactoryConcept) factory for the H<->C row-major transpose path.
struct TransposeHCRMProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
