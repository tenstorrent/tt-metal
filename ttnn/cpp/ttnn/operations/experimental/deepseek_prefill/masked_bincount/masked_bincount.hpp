// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount {

ttnn::Tensor masked_bincount(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_mask,
    uint32_t n_routed_experts,
    uint32_t num_experts_per_token);

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount

namespace ttnn {
using operations::experimental::deepseek_prefill::masked_bincount::masked_bincount;
}  // namespace ttnn
