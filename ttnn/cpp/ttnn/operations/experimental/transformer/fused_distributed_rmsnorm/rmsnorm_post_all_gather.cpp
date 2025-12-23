// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather.hpp"

#include "ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/fused_rmsnorm_post_all_gather_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor ExecuteFusedRMSNormPostAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& stats,
    float epsilon,
    uint32_t num_heads,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& transformation_mat,
    const std::optional<const ttnn::Tensor>& rope_cos,
    const std::optional<const ttnn::Tensor>& rope_sin,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType>& dtype) {
    auto arch = input_tensor.device()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, false, true, false);

    return ttnn::prim::fused_rmsnorm_post_all_gather(
        input_tensor,
        stats,
        epsilon,
        num_heads,
        weight,
        transformation_mat,
        rope_cos,
        rope_sin,
        memory_config.value_or(input_tensor.memory_config()),
        kernel_config_val,
        dtype);
}

}  // namespace ttnn::operations::experimental::transformer
