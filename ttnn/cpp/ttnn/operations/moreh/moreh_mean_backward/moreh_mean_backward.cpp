// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_backward.hpp"

#include <algorithm>

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/device/moreh_mean_backward_device_operation.hpp"

namespace ttnn {

Tensor moreh_mean_backward(
    const Tensor& output_grad,
    std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<ttnn::Shape>& input_grad_shape,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto output_grad_rank = output_grad.logical_shape().rank();
    auto input_grad_rank = output_grad_rank;
    if (input_grad_shape.has_value()) {
        input_grad_rank = input_grad_shape->rank();
    } else if (input_grad.has_value()) {
        input_grad_rank = input_grad->logical_shape().rank();
    } else if (!keepdim) {
        if (!dim.has_value()) {
            // do nothing
        } else if (std::holds_alternative<int64_t>(dim.value())) {
            input_grad_rank += 1;
        } else {
            auto dims = std::get<ttsl::SmallVector<int64_t>>(dim.value());
            input_grad_rank += dims.size();
        }
    }
    ttsl::SmallVector<int64_t> dims = ttnn::operations::get_dim(dim, input_grad_rank);
    // The program broadcasts output_grad along the reduced dims in their keepdim positions. A keepdim=false
    // output_grad that dropped them lays its tiles out differently once a reduced dim is one of the last two,
    // so put the reduced dims back as size 1 first.
    if (!keepdim && output_grad_rank < input_grad_rank) {
        const auto& output_grad_shape = output_grad.logical_shape();
        ttsl::SmallVector<uint32_t> keepdim_shape;
        for (int64_t i = 0, src = 0; i < static_cast<int64_t>(input_grad_rank); ++i) {
            const bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
            keepdim_shape.push_back(reduced ? 1 : output_grad_shape[src++]);
        }
        return ttnn::prim::moreh_mean_backward(
            ttnn::reshape(output_grad, ttnn::Shape(keepdim_shape)),
            dims,
            true,
            input_grad_shape,
            input_grad,
            memory_config,
            compute_kernel_config);
    }
    return ttnn::prim::moreh_mean_backward(
        output_grad, dims, keepdim, input_grad_shape, input_grad, memory_config, compute_kernel_config);
}

}  // namespace ttnn
