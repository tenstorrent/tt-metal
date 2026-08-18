// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reduce_affine_transforms_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct ReduceAffineTransformsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ReduceAffineTransformsParams&, const ReduceAffineTransformsInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
