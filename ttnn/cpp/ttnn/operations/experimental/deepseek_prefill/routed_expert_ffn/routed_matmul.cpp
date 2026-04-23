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
    const std::optional<ttnn::Tensor>& max_expert_iter,
    uint32_t curr_expert_iter) {
    TT_FATAL(program_config.has_value(), "routed_matmul: program_config is required (no auto-select path)");
    TT_FATAL(
        compute_kernel_config.has_value(), "routed_matmul: compute_kernel_config is required (no auto-select path)");
    TT_FATAL(max_expert_iter.has_value(), "routed_matmul: max_expert_iter is required");

    auto result = ttnn::prim::routed_matmul(
        input_tensor_a,
        input_tensor_b,
        max_expert_iter.value(),
        curr_expert_iter,
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
