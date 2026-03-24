// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount.hpp"
#include "device/masked_bincount_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount {

ttnn::Tensor masked_bincount(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& expert_mask, uint32_t n_routed_experts) {
    return ttnn::prim::masked_bincount(input_tensor, expert_mask, n_routed_experts);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount
