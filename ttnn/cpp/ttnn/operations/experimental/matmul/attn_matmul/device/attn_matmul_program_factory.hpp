// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "attn_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct AttnMatmulProgramFactory {
    // Metal 2.0 ProgramSpecFactoryConcept: one ProgramSpec stamped across the mesh. The in0 DFB's
    // capacity (Kt entries) depends on the input shape; the default reflection hash keys the program
    // cache on the full tensor specs, so each unique Kt keeps its own cache entry. On a cache hit the
    // framework refreshes only the three tensor bindings (input_a, input_b, output).
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const AttnMatmulParams& operation_attributes, const AttnMatmulInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
