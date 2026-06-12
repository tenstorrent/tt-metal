// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm.hpp"

namespace ttnn {

Tensor moreh_bmm(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // TODO(nuked-op matmul): restore real call (was ttnn::moreh_matmul(input, mat2, ...))
    (void)mat2;
    (void)output;
    (void)memory_config;
    (void)compute_kernel_config;
    Tensor out = input;
    return out;
}

}  // namespace ttnn
