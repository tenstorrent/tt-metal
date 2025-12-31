// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "ema.hpp"

#include "device/ema_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::reduction::accumulation {

Tensor ema(
    const Tensor& input_tensor,
    const float& alpha,
    std::optional<Tensor> optional_out,
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::operations::reduction::ema::EmaDeviceOperation;
    auto grid_size = core_grid.value_or(CoreGrid(0, 0)).to_CoreCoord();
    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
    auto kernel_config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .alpha = alpha,
            .grid_size = grid_size,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = kernel_config,
        },
        OperationType::tensor_args_t{
            .input = input_tensor,
            .optional_output_tensor = std::move(optional_out),
        });
}

}  // namespace ttnn::operations::reduction::accumulation
