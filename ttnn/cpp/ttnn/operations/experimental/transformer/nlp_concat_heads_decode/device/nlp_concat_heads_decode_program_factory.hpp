// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "nlp_concat_heads_decode_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsDecodeProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NlpConcatHeadsDecodeParams& operation_attributes,
        const NlpConcatHeadsDecodeInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
