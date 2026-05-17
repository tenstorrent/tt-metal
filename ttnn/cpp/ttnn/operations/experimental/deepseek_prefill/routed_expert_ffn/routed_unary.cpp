// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_unary.hpp"
#include "device/routed_unary_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_unary(
    const ttnn::Tensor& input,
    const ttnn::operations::unary::EltwiseUnaryWithParam& op,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_token_counts,
    uint32_t local_expert_idx,
    uint32_t curr_expert_iter,
    uint32_t expert_iter_length,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> optional_output_tensor) {
    TT_FATAL(
        compute_kernel_config.has_value(), "routed_unary: compute_kernel_config is required (no auto-select path)");
    TT_FATAL(expert_iter_length > 0, "routed_unary: expert_iter_length must be > 0");

    return ttnn::prim::routed_unary(
        input,
        op,
        global_expert_idx_table,
        expert_token_counts,
        local_expert_idx,
        curr_expert_iter,
        expert_iter_length,
        compute_kernel_config.value(),
        /*output_memory_config=*/std::nullopt,
        std::move(optional_output_tensor),
        /*output_dtype=*/std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
