// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "ema.hpp"

#include "device/ema_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::reduction::accumulation {

Tensor EmaOperation::invoke(
    const Tensor& input_tensor,
    const float& alpha,
    std::optional<Tensor> optional_out,
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto grid_size = core_grid.value_or(CoreGrid(0, 0)).to_CoreCoord();
    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
    auto kernel_config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::prim::ema_device(
        input_tensor, alpha, grid_size, output_mem_config, kernel_config, std::move(optional_out));
}

}  // namespace ttnn::operations::reduction::accumulation
