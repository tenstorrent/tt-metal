// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "create_qkv_heads_decode_device_operation_types.hpp"

#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct CreateQKVHeadsDecodeShardedProgramFactory {
    using operation_attributes_t = CreateQkvHeadsDecodeParams;
    using tensor_args_t = CreateQkvHeadsDecodeInputs;
    using tensor_return_value_t = std::vector<Tensor>;

    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);
};

}  // namespace ttnn::experimental::prim
