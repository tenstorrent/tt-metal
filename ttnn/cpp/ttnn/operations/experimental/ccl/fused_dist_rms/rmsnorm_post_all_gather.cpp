// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather.hpp"

#include "ttnn/operations/experimental/ccl/fused_dist_rms/device/rmsnorm_post_all_gather_op.hpp"

namespace ttnn::operations::experimental::ccl {

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
    return tt::tt_metal::operation::run(
               FusedRMSNormPostAllGather{
                   .eps = epsilon,
                   .num_heads = num_heads,
                   .memory_config = memory_config.value_or(input_tensor.memory_config()),
                   .compute_kernel_config = kernel_config_val,
                   .dtype = dtype},
               {input_tensor, stats},
               {weight, transformation_mat, rope_cos, rope_sin})
        .at(0);
}

}  // namespace ttnn::operations::experimental::ccl
