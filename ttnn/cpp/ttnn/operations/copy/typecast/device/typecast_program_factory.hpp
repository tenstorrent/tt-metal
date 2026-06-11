// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::prim {

// Metal 2.0 typecast interleaved factory — the degenerate ProgramSpecFactoryConcept. Every run-arg is a
// pure function of the input/output layout and the compute grid (no per-call dynamic scalar), so the
// simplest concept applies: create_program_artifacts returns the spec PLUS all run-args; on a cache hit
// the framework just refreshes the tensor bindings (UpdateTensorArgs).
struct TypecastProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

// Metal 2.0 typecast sub-core-grid factory — degenerate ProgramSpecFactoryConcept (same rationale).
struct TypecastSubgridProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
