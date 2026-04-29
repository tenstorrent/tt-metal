// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bf16_to_fp8.hpp"

#include "device/bf16_to_fp8_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8 {

ttnn::Tensor bf16_to_fp8(const ttnn::Tensor& input_tensor) { return ttnn::prim::prefill_bf16_to_fp8(input_tensor); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::bf16_to_fp8
