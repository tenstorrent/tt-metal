// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

// Metal 2.0 (ProgramSpecFactoryConcept) factory for the single-core, TILE-layout pad path.
struct PadTileCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
