// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "move_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// Program factory for MULTI_CORE_OVERLAP strategy (Metal 2.0 / MetalV2FactoryConcept).
struct MoveOverlapProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MoveOperationAttributes& operation_attributes,
        const MoveTensorArgs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
