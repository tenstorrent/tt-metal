// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear.hpp"

#include <mutex>

#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn {

Tensor moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    static std::once_flag deprecation_warned;
    std::call_once(deprecation_warned, [] {
        log_warning(tt::LogOp, "ttnn.moreh_linear is deprecated; use ttnn.linear with transpose_b=True.");
    });
    return ttnn::linear(
        input,
        weight,
        bias,
        /*transpose_a=*/false,
        /*transpose_b=*/true,
        output.has_value() ? memory_config : memory_config.value_or(input.memory_config()),
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        output);
}

}  // namespace ttnn
