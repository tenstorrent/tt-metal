// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QkvCausalConv1dSiluParams&, const QkvCausalConv1dSiluInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
