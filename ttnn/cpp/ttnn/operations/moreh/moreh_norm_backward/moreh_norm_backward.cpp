// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward.hpp"

#include <cmath>
#include "device/moreh_norm_backward_device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn {

Tensor moreh_norm_backward(
    const Tensor& input,
    const Tensor& output,
    const Tensor& output_grad,
    float p,
    const std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>>& dim,
    bool keepdim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (p == 0.0f) {
        if (input_grad.has_value()) {
            return ttnn::zeros_like(input_grad.value());
        }
        return ttnn::zeros_like(input, std::nullopt, std::nullopt, std::nullopt, memory_config.value_or(input.memory_config()));
    }

    if (std::isinf(p)) {
        Tensor abs_input = ttnn::abs(input);
        Tensor expanded_output = output;
        if (!keepdim && dim.has_value()) {
            ttsl::SmallVector<int64_t> dims = ttnn::operations::get_dim(dim, input.logical_shape().rank());
            auto new_shape = input.logical_shape();
            for (auto d : dims) {
                new_shape[d] = 1;
            }
            expanded_output = ttnn::reshape(output, new_shape);
        }
        Tensor mask = ttnn::eq(abs_input, expanded_output);
        Tensor sign_input = ttnn::sign(input);
        Tensor subgrad = ttnn::multiply(sign_input, mask);
        Tensor result = ttnn::multiply(subgrad, output_grad);
        if (input_grad.has_value()) {
            ttnn::copy(result, input_grad.value());
            return input_grad.value();
        }
        return result;
    }

    return ttnn::prim::moreh_norm_backward(
        input, output, output_grad, p, dim, keepdim, input_grad, memory_config, compute_kernel_config);
}

}  // namespace ttnn
