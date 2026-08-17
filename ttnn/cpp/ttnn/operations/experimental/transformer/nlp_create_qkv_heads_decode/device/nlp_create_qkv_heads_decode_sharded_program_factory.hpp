// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "nlp_create_qkv_heads_decode_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct NLPCreateQKVHeadsDecodeShardedProgramFactory {
    using operation_attributes_t = NlpCreateQkvHeadsDecodeParams;
    using tensor_args_t = NlpCreateQkvHeadsDecodeInputs;
    using tensor_return_value_t = std::vector<Tensor>;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);
};

}  // namespace ttnn::experimental::prim
