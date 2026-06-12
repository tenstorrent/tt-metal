// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear.hpp"

namespace ttnn {

Tensor moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // TODO(nuked-op matmul): restore real call (was ttnn::moreh_matmul(input, weight, ...))
    (void)weight;
    (void)bias;
    (void)output;
    (void)memory_config;
    (void)compute_kernel_config;
    Tensor out = input;
    return out;
}

}  // namespace ttnn
