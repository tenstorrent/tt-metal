// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward.hpp"

#include <algorithm>

#include "device/moreh_norm_backward_device_operation.hpp"
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
    // The program broadcasts output and output_grad along the reduced dims in their keepdim positions. A
    // keepdim=false tensor that dropped them lays its tiles out differently once a reduced dim is one of the last
    // two, so put the reduced dims back as size 1 first.
    const auto& input_shape = input.logical_shape();
    if (!keepdim && dim.has_value() &&
        (output.logical_shape().rank() < input_shape.rank() ||
         output_grad.logical_shape().rank() < input_shape.rank())) {
        const auto dims = ttnn::operations::get_dim(dim, input_shape.rank());
        ttsl::SmallVector<uint32_t> keepdim_shape;
        for (int64_t i = 0; i < static_cast<int64_t>(input_shape.rank()); ++i) {
            const bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
            keepdim_shape.push_back(reduced ? 1 : input_shape[i]);
        }
        const ttnn::Shape shape(keepdim_shape);
        return ttnn::prim::moreh_norm_backward(
            input,
            ttnn::reshape(output, shape),
            ttnn::reshape(output_grad, shape),
            p,
            dim,
            true,
            input_grad,
            memory_config,
            compute_kernel_config);
    }
    return ttnn::prim::moreh_norm_backward(
        input, output, output_grad, p, dim, keepdim, input_grad, memory_config, compute_kernel_config);
}

}  // namespace ttnn
