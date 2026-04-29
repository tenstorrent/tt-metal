// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8 {

ttnn::Tensor bf16_to_fp8(const ttnn::Tensor& input_tensor);

}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8

namespace ttnn {
using operations::experimental::deepseek_prefill::bf16_to_fp8::bf16_to_fp8;
}  // namespace ttnn
