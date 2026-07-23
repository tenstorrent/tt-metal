// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "nlp_create_qkv_heads_decode_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct NLPCreateQKVHeadsDecodeInterleavedProgramFactory {
    using operation_attributes_t = NlpCreateQkvHeadsDecodeParams;
    using tensor_args_t = NlpCreateQkvHeadsDecodeInputs;
    using tensor_return_value_t = std::vector<Tensor>;

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);
};

}  // namespace ttnn::experimental::prim
