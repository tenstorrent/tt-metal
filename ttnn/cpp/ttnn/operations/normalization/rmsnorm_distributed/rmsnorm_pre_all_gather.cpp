// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pre_all_gather.hpp"

#include "ttnn/device.hpp"

namespace ttnn {

ttnn::Tensor rms_norm_pre_all_gather(
    const ttnn::Tensor& input_tensor,
    const DataType dtype,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<bool>& use_2d_core_grid) {
    (void)dtype;
    (void)residual_input_tensor;
    (void)compute_kernel_config;
    (void)program_config;
    (void)memory_config;
    (void)use_2d_core_grid;
    (void)ttnn::GetDefaultDevice();
    // TODO(nuked-op layernorm): restore real distributed pre-all-gather RMSNorm path.
    return input_tensor;
}

}  // namespace ttnn
