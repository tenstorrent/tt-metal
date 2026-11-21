// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b_spec/swiglu_decode/swiglu_decode.hpp"

#include "ttnn/operations/experimental/llama31_8b_spec/swiglu_decode/device/swiglu_decode_device_operation.hpp"

namespace ttnn::experimental::llama31_8b_spec {

ttnn::Tensor swiglu_decode(const ttnn::Tensor& gate, const ttnn::Tensor& up) {
    return ttnn::prim::llama31_8b_spec_swiglu_decode(gate, up);
}

}  // namespace ttnn::experimental::llama31_8b_spec
