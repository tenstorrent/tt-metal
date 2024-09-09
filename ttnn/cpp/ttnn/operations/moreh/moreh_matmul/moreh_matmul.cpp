// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul.hpp"

#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_matmul {
Tensor MorehMatmul::invoke(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<Tensor>& output,
    const std::optional<const Tensor> bias,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::moreh_matmul(
        input, other, transpose_input, transpose_other, output, bias, output_mem_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_matmul
