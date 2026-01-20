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
    std::optional<unary::UnaryWithParam> fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    // TODO: Full fusion will modify minimal_matmul kernels to compute addcmul inline
    // For now, we only call minimal_matmul and ignore the addcmul parameters
    // The full implementation will compute: output = addcmul_input_tensor1 + (scalar * matmul_output *
    // addcmul_input_tensor2) where matmul_output = minimal_matmul(matmul_input_tensor, matmul_weight_tensor)

    // Unused parameters in skeleton implementation
    (void)scalar;
    (void)addcmul_input_tensor1;
    (void)addcmul_input_tensor2;

    // Call minimal_matmul with the provided parameters
    return ttnn::prim::minimal_matmul(
        matmul_input_tensor,
        matmul_weight_tensor,
        bias_tensor,
        std::move(fused_activation),
        config,
        memory_config,
        dtype,
        compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::transformer
