// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_dot/moreh_dot.hpp"
#include "ttnn/operations/moreh/moreh_matmul/device/moreh_matmul_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_matmul {

inline bool is_dot_forward(const Tensor& input, const Tensor& other, bool transpose_input, bool transpose_other) {
    // TODO: non-4d support for dot.
    if (input.get_legacy_shape().rank() != 4 || other.get_legacy_shape().rank() != 4) {
        return false;
    }

    if (transpose_input || transpose_other) {
        return false;
    }

    return tt::operations::primary::is_1d_tensor(input) && tt::operations::primary::is_1d_tensor(other) &&
           tt::operations::primary::is_same_shape(input, other);
}

Tensor MorehMatmul::invoke(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<Tensor>& output,
    const std::optional<const Tensor> bias,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    if (is_dot_forward(input, other, transpose_input, transpose_other)) {
        return ttnn::moreh_dot(input, other, output, input.get_dtype(), memory_config, compute_kernel_config);
    }
    return ttnn::prim::moreh_matmul(
        input, other, transpose_input, transpose_other, output, bias, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_matmul
