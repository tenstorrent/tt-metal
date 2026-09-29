// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Gated-DeltaNet gates from the columns of the [g|a|b] projection output, in ONE op:
//   beta = bf16(sigmoid(b)) * beta_scale          (bf16, then widened to fp32)
//   g    = a_neg * bf16(softplus(bf16(a + dt_bias)))   (bf16, then widened to fp32)
// Bit-identical to the binary_ng chain multiply(sigmoid act) / add(softplus act) / multiply.
std::tuple<ttnn::Tensor, ttnn::Tensor> gdn_gates(
    const ttnn::Tensor& gab,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& a_neg,
    uint32_t a_col_offset,
    uint32_t b_col_offset,
    uint32_t num_heads,
    float beta_scale = 1.0f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
