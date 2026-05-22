// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nlp_create_qkv_heads_falcon7b_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsFalcon7BProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NlpCreateQkvHeadsFalcon7bParams& operation_attributes,
        const Tensor& tensor_args,
        NlpCreateQkvHeadsFalcon7bResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
