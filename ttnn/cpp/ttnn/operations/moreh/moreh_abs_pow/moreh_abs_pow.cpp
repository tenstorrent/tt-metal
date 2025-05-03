// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow.hpp"

#include "ttnn/operations/moreh/moreh_abs_pow/device/moreh_abs_pow_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {
Tensor MorehAbsPow::invoke(
    const Tensor& input,
    const float p,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_abs_pow(input, p, output, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_abs_pow
