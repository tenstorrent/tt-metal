// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_matmul.hpp"
#include "device/routed_matmul_device_operation.hpp"

#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    const std::optional<const ttnn::operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const ttnn::Activation>& activation,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<ttnn::Tensor> optional_output_tensor,
    const std::optional<ttnn::Tensor>& global_expert_idx_table,
    const std::optional<ttnn::Tensor>& expert_token_counts,
    uint32_t local_expert_idx,
    uint32_t curr_expert_iter,
    uint32_t expert_iter_length) {
    TT_FATAL(program_config.has_value(), "routed_matmul: program_config is required (no auto-select path)");
    TT_FATAL(
        compute_kernel_config.has_value(), "routed_matmul: compute_kernel_config is required (no auto-select path)");
    TT_FATAL(global_expert_idx_table.has_value(), "routed_matmul: global_expert_idx_table is required");
    TT_FATAL(expert_token_counts.has_value(), "routed_matmul: expert_token_counts is required");
    TT_FATAL(expert_iter_length > 0, "routed_matmul: expert_iter_length must be > 0");

    auto result = ttnn::prim::routed_matmul(
        input_tensor_a,
        input_tensor_b,
        global_expert_idx_table.value(),
        expert_token_counts.value(),
        local_expert_idx,
        curr_expert_iter,
        expert_iter_length,
        program_config.value(),
        compute_kernel_config.value(),
        memory_config,
        optional_output_tensor,
        dtype);

    // Mirror non-routed matmul: when activation is passed via this parameter
    // (not via program_config.fused_activation), append it as a separate
    // unary op on the host-side trace rather than fusing into the matmul kernel.
    if (auto fused = ttnn::operations::matmul::get_fused_activation(activation); fused.has_value()) {
        result = ttnn::unary_chain(result, {fused.value()}, result.memory_config(), optional_output_tensor);
    }
    return result;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
