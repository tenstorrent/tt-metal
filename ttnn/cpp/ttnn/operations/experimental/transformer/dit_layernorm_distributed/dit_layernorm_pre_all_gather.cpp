// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_pre_all_gather.hpp"

#include "device/dit_layernorm_pre_all_gather_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor ExecuteDitLayerNormPreAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const DataType dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config) {
    auto arch = input_tensor.device()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

    return ttnn::prim::dit_layernorm_pre_all_gather(
        input_tensor, dtype, kernel_config_val, memory_config.value_or(input_tensor.memory_config()));
}

}  // namespace ttnn::operations::experimental::transformer
