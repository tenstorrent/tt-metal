// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsVitProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NlpCreateQkvHeadsVitParams& operation_attributes,
        const NlpCreateQkvHeadsVitInputs& tensor_args,
        NlpCreateQkvHeadsVitResult& output);
};

}  // namespace ttnn::experimental::prim
