// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm.hpp"

// TODO(nuke-op matmul): restore real call — moreh_matmul deleted with the matmul op

namespace ttnn {

Tensor moreh_bmm(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // TODO(nuked-op matmul): restore real call — ttnn::moreh_matmul deleted; passthrough first tensor arg.
    (void)mat2;
    (void)output;
    (void)memory_config;
    (void)compute_kernel_config;
    return input;
}

}  // namespace ttnn
