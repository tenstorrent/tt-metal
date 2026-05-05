// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract {

ttnn::Tensor extract(
    const ttnn::Tensor& global_tensor,
    const ttnn::Tensor& start,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    uint32_t max_dispatched_tokens_per_expert);

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract

namespace ttnn {
using operations::experimental::deepseek_prefill::extract::extract;
}  // namespace ttnn
