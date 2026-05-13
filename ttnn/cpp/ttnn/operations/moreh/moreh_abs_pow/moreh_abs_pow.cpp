// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow.hpp"

#include "ttnn/operations/moreh/moreh_abs_pow/device/moreh_abs_pow_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor moreh_abs_pow(
    const Tensor& input,
    const float p,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_OP_SCOPE("ttnn::moreh_abs_pow");
    return ttnn::prim::moreh_abs_pow(input, p, output, memory_config, compute_kernel_config);
}

}  // namespace ttnn
