// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather.hpp"

#include "ttnn/device.hpp"

namespace ttnn {

ttnn::Tensor rms_norm_post_all_gather(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& stats,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    const std::optional<const DataType>& dtype,
    const std::optional<bool>& use_2d_core_grid) {
    (void)stats;
    (void)epsilon;
    (void)weight;
    (void)bias;
    (void)memory_config;
    (void)compute_kernel_config;
    (void)program_config;
    (void)dtype;
    (void)use_2d_core_grid;
    (void)ttnn::GetDefaultDevice();
    // TODO(nuked-op layernorm): restore real distributed post-all-gather RMSNorm path.
    return input_tensor;
}

}  // namespace ttnn
