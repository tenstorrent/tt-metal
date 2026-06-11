// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_concat_heads_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::experimental::prim {

// Metal 2.0 nlp_concat_heads factory — the degenerate ProgramSpecFactoryConcept.
//
// The op has no per-call dynamic scalar (no seed-like value): every run-arg is a pure function of the
// input/output layout and the compute grid, so there is nothing to vary between two dispatches that
// share a cache entry. So we use the simplest concept: create_program_artifacts returns the spec PLUS
// ALL run-args bundled together; on a cache hit the framework just refreshes the tensor bindings
// (UpdateTensorArgs). No extract_immutable_info, no create_per_enqueue_args, no custom hash — the
// framework hashes the generated ProgramSpec (and tensor specs) for the cache key.
struct NLPConcatHeadsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const NlpConcatHeadsParams& operation_attributes, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::experimental::prim
