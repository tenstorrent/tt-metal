// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tilize_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Metal 2.0 tilize multi-core default factory — the degenerate ProgramSpecFactoryConcept.
//
// Every run-arg is a pure function of the input/output layout and the compute grid (work split, tile/page
// start ids); nothing varies between two dispatches that share a cache entry. So we use the simplest
// concept: create_program_artifacts returns the spec PLUS ALL run-args bundled together; on a cache hit
// the framework just refreshes the tensor bindings (UpdateTensorArgs). No extract_immutable_info, no
// create_per_enqueue_args, no custom hash.
struct TilizeMultiCoreDefaultProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
