// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm.hpp"

#include "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"

namespace ttnn::operations::moreh::moreh_bmm {
Tensor MorehBmm::invoke(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::moreh_matmul(input, mat2, false, false, output, std::nullopt, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_bmm
