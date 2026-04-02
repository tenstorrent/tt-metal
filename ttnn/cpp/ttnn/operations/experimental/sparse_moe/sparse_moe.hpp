// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

Tensor sparse_moe_expert(
    const Tensor& input,
    const Tensor& expert_gu,
    const Tensor& expert_dw,
    const Tensor& expert_mask,
    uint32_t num_experts,
    uint32_t expert_inter_dim,
    uint32_t hidden_dim,
    uint32_t batch_size);

}  // namespace ttnn::experimental
