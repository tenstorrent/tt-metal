// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdn_decode_step_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct GdnDecodeStepProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GdnDecodeStepParams& attrs, const GdnDecodeStepInputs& in, Tensor& output);
};

}  // namespace ttnn::experimental::prim
