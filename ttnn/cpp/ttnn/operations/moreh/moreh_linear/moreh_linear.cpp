// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear.hpp"

#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_OP_SCOPE("ttnn::moreh_linear");
    return ttnn::moreh_matmul(input, weight, false, true, output, bias, memory_config, compute_kernel_config);
}

}  // namespace ttnn
