// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm.hpp"

#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor moreh_bmm(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_OP_SCOPE("ttnn::moreh_bmm");
    return ttnn::moreh_matmul(input, mat2, false, false, output, std::nullopt, memory_config, compute_kernel_config);
}

}  // namespace ttnn
