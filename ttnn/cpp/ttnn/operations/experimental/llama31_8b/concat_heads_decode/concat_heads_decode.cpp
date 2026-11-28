// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/concat_heads_decode/concat_heads_decode.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/concat_heads_decode/device/concat_heads_decode_device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

ttnn::Tensor concat_heads_decode(const ttnn::Tensor& input_tensor, uint32_t num_heads) {
    using namespace ttnn::prim::llama31_8b_concat_heads_decode;
    return ttnn::device_operation::launch<ConcatHeadsDecodeDeviceOperation>(
        ConcatHeadsDecodeParams{.num_heads = num_heads}, ConcatHeadsDecodeInputs{.input = input_tensor});
}

}  // namespace ttnn::experimental::llama31_8b
