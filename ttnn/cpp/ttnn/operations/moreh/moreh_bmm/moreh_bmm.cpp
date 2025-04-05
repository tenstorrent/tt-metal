// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm.hpp"

#include <algorithm>

#include "cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::moreh::moreh_bmm {
Tensor MorehBMM::invoke(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::moreh_matmul(input, mat2, false, false, output, std::nullopt, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_bmm
