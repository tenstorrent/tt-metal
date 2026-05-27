// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tilize_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Metal 2.0 port: satisfies ProgramSpecFactoryConcept.
// See METAL2_PORT_PLAN.md / METAL2_PORT_REPORT.md alongside the op directory.
struct TilizeMultiCoreWidthShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
