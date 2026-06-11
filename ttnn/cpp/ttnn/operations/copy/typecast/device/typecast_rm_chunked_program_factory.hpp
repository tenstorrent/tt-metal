// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::prim {

// Metal 2.0 typecast row-major chunked factory — the degenerate ProgramSpecFactoryConcept. Every run-arg
// is a pure function of layout + grid, so create_program_artifacts returns the spec PLUS all run-args;
// the cache hit refreshes only the tensor bindings (UpdateTensorArgs).
struct TypecastRowMajorChunkedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
