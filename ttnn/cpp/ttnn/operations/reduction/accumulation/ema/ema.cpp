// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "ema.hpp"
#include "ema_op.hpp"

namespace ttnn::operations::reduction::accumulation {

Tensor EmaOperation::invoke(
    const Tensor& input_tensor,
    const float& alpha,
    std::optional<Tensor> optional_out,
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return tt::tt_metal::operation::run(
               Ema{
                   .alpha = alpha,
                   .grid_size = core_grid.value_or(CoreGrid(0, 0)).to_CoreCoord(),
                   .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                   .compute_kernel_config = init_device_compute_kernel_config(
                       input_tensor.device()->arch(),
                       compute_kernel_config,
                       /*math_fidelity=*/MathFidelity::HiFi4,
                       /*default_approx_mode=*/false,
                       /*fp32_acc=*/true),
               },
               {input_tensor},
               {},
               {std::move(optional_out)})
        .at(0);
}

}  // namespace ttnn::operations::reduction::accumulation
