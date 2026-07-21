// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsSegformerProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NlpCreateQkvHeadsSegformerParams& operation_attributes,
        const NlpCreateQkvHeadsSegformerInputs& tensor_args,
        NlpCreateQkvHeadsSegformerResult& output);
};

}  // namespace ttnn::experimental::prim
