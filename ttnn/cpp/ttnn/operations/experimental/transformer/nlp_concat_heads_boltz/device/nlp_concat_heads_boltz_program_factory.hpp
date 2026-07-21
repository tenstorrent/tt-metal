// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "nlp_concat_heads_boltz_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct NLPConcatHeadsBoltzProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NLPConcatHeadsBoltzParams& operation_attributes,
        const NLPConcatHeadsBoltzInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
