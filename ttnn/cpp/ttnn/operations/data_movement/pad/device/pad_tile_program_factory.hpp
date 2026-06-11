// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

// Metal 2.0 pad (tiled, single-core) factory — the degenerate ProgramSpecFactoryConcept.
//
// pad_value is a per-call scalar but structural (baked as a named runtime arg derived from the
// attributes, which enter the cache key). Every other run-arg is a pure function of the input/output
// layout. create_program_artifacts returns the spec PLUS ALL run-args; on a cache hit the framework
// refreshes the tensor bindings (UpdateTensorArgs). No extract_immutable_info, no
// create_per_enqueue_args, no custom hash.
struct PadTileCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);
};
}  // namespace ttnn::prim
