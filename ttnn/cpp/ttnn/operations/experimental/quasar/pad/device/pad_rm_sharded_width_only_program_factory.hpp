// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Metal 2.0 (MetalV2FactoryConcept) factory for the width-only sharded ROW_MAJOR (stickwise) pad path.
struct PadRmShardedWidthOnlyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);
};
}  // namespace ttnn::prim::qsr
