// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_minimal_matmul_addcmul_fused.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor ExecuteDitMinimalMatmulAddcmulFused::invoke(
    const ttnn::Tensor& matmul_input_tensor,
    const ttnn::Tensor& matmul_weight_tensor,
    float scalar,
    const ttnn::Tensor& addcmul_input_tensor1,
    const ttnn::Tensor& addcmul_input_tensor2,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    // Delegates to minimal_matmul with fused addcmul (ternary) parameters.
    // Formula: output = addcmul_input_tensor1 + (scalar * matmul_output * addcmul_input_tensor2).
    auto outputs = ttnn::prim::minimal_matmul(
        matmul_input_tensor,
        matmul_weight_tensor,
        bias_tensor,
        std::nullopt,  // no fused_activation for dit_minimal_matmul_addcmul_fused
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        1,                       // no splitting
        -1,                      // dim
        scalar,                  // fused_ternary_scalar
        addcmul_input_tensor1,   // fused_ternary_input_a (residual/base, full shape)
        addcmul_input_tensor2);  // fused_ternary_input_b (gate/multiplier, broadcast)

    TT_FATAL(outputs.size() == 1, "Expected single output from minimal_matmul, got {}", outputs.size());
    return outputs[0];
}

}  // namespace ttnn::operations::experimental::transformer
