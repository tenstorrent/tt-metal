// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_matmul.hpp"

#include <mutex>

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_dot/moreh_dot.hpp"

namespace ttnn::operations::moreh::moreh_matmul {

inline bool is_dot_forward(const Tensor& input, const Tensor& other, bool transpose_input, bool transpose_other) {
    // TODO: non-4d support for dot.
    if (input.padded_shape().rank() != 4 || other.padded_shape().rank() != 4) {
        return false;
    }

    if (transpose_input || transpose_other) {
        return false;
    }

    return is_1d_tensor(input) && is_1d_tensor(other) && is_same_shape(input, other);
}

}  // namespace ttnn::operations::moreh::moreh_matmul

namespace ttnn {

Tensor moreh_matmul(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input,
    bool transpose_other,
    const std::optional<Tensor>& output,
    const std::optional<const Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    static std::once_flag deprecation_warned;
    std::call_once(deprecation_warned, [] {
        log_warning(tt::LogOp, "ttnn.moreh_matmul is deprecated; use ttnn.matmul (or ttnn.linear with a bias).");
    });
    if (operations::moreh::moreh_matmul::is_dot_forward(input, other, transpose_input, transpose_other)) {
        return ttnn::moreh_dot(input, other, output, input.dtype(), memory_config, compute_kernel_config);
    }
    return ttnn::linear(
        input,
        other,
        bias,
        transpose_input,
        transpose_other,
        output.has_value() ? memory_config : memory_config.value_or(input.memory_config()),
        /*dtype=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt,
        compute_kernel_config,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt,
        output);
}

}  // namespace ttnn
