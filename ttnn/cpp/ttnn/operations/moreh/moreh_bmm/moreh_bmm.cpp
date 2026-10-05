// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm.hpp"

#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn {

Tensor moreh_bmm(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::matmul(
        input,
        mat2,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
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
