// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::prim {

// Metal 2.0 CN-transpose factory — the degenerate ProgramSpecFactoryConcept.
//
// CN transpose has no per-call dynamic scalar (no seed-like value): every run-arg is a pure function of
// the input/output layout and the compute grid, so there is nothing to vary between two dispatches that
// share a cache entry. create_program_artifacts returns the spec PLUS all run-args bundled together; on
// a cache hit the framework just refreshes the tensor bindings (UpdateTensorArgs). No
// extract_immutable_info, no create_per_enqueue_args, no custom hash — the framework hashes the
// generated ProgramSpec (and tensor specs) for the cache key.
struct TransposeCNProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
