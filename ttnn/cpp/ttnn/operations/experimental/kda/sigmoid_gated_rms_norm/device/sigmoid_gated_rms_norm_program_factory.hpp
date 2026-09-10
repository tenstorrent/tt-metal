// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sigmoid_gated_rms_norm_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct SigmoidGatedRmsNormProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SigmoidGatedRmsNormParams&, const SigmoidGatedRmsNormInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
