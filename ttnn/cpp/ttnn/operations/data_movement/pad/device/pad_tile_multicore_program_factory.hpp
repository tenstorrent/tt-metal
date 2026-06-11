// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "pad_device_operation_types.hpp"

namespace ttnn::prim {

// Metal 2.0 pad (tiled, multi-core) factory — the degenerate ProgramSpecFactoryConcept.
//
// pad_value is a per-call scalar but is structural: it is baked into the kernel as a compile-time arg,
// so it enters the cache key via the operation attributes. Every other run-arg is a pure function of the
// input/output layout and the compute grid, so there is nothing to vary between two dispatches that
// share a cache entry. We use the simplest concept: create_program_artifacts returns the spec PLUS ALL
// run-args bundled together; on a cache hit the framework refreshes the tensor bindings
// (UpdateTensorArgs). No extract_immutable_info, no create_per_enqueue_args, no custom hash.
struct PadTileMulticoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output);
};
}  // namespace ttnn::prim
