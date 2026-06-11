// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_concat_heads_decode_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::experimental::prim {

// Metal 2.0 factory — degenerate ProgramSpecFactoryConcept (subcoregrids variant). Same reasoning as
// NLPConcatHeadsDecodeProgramFactory: nothing varies per call, so the full run-args ship with the spec.
struct NLPConcatHeadsDecodeSubcoregridsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const NlpConcatHeadsDecodeParams& operation_attributes,
        const NlpConcatHeadsDecodeInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
