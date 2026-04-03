// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "sparse_moe.hpp"
#include "device/sparse_moe_device_operation.hpp"

namespace ttnn::experimental {

Tensor sparse_moe_expert(
    const Tensor& input,
    const Tensor& expert_gu,
    const Tensor& expert_dw,
    const Tensor& expert_mask,
    uint32_t num_experts,
    uint32_t expert_inter_dim,
    uint32_t hidden_dim,
    uint32_t batch_size) {
    return ttnn::device_operation::launch<ttnn::operations::experimental::sparse_moe::SparseMoeExpertOperation>(
        ttnn::operations::experimental::sparse_moe::SparseMoeExpertOperation::operation_attributes_t{
            .num_experts = num_experts,
            .expert_inter_dim = expert_inter_dim,
            .hidden_dim = hidden_dim,
            .batch_size = batch_size,
        },
        ttnn::operations::experimental::sparse_moe::SparseMoeExpertOperation::tensor_args_t{
            .input = input,
            .expert_gu = expert_gu,
            .expert_dw = expert_dw,
            .expert_mask = expert_mask,
        });
}

}  // namespace ttnn::experimental
