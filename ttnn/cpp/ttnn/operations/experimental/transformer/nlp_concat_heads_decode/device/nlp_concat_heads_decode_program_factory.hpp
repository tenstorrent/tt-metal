// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_concat_heads_decode_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::experimental::prim {

// Metal 2.0 factory — degenerate ProgramSpecFactoryConcept.
//
// This op has no per-call dynamic scalar: everything the program needs is a pure function of the input
// shard layout + output shard layout (the head-offset table, the input NOC coordinate arrays, the shape
// scalars). So create_program_artifacts returns the spec PLUS the complete run-args, and the cache-hit
// path only refreshes the tensor bindings (UpdateTensorArgs). No extract_immutable_info, no
// create_per_enqueue_args, no custom hash.
struct NLPConcatHeadsDecodeProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const NlpConcatHeadsDecodeParams& operation_attributes,
        const NlpConcatHeadsDecodeInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
