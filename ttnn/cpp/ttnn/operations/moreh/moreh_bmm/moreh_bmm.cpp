// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm.hpp"

namespace ttnn::operations::moreh::moreh_bmm {
Tensor MorehBmm::invoke(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_matmul(input, mat2, false, false, output, std::nullopt, output_memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_bmm
