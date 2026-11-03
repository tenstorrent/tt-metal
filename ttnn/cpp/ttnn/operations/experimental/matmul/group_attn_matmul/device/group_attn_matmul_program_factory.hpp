// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "group_attn_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct GroupAttnMatmulProgramFactory {
    // Metal 2.0 ProgramSpecFactoryConcept: one ProgramSpec stamped across the mesh. Several DFB
    // capacities depend on (KV_HEADS, Mt, Kt, Nt) computed from the input shapes; the default
    // reflection hash keys the program cache on the full tensor specs, so each unique sizing keeps
    // its own cache entry. On a cache hit the framework refreshes only the three tensor bindings
    // (input_a, input_b, output), which also re-attaches the sharded operands' borrowed DFBs.
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GroupAttnMatmulParams& operation_attributes,
        const GroupAttnMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
