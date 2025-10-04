// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_pybind.hpp"

#include "softmax_backward.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::normalization::detail {
namespace py = pybind11;

// Softmax backward operation base
void bind_normalization_softmax_backward_operation(py::module& module) {
    const auto doc =
        R"doc(
            Computes the backward pass (gradient) of the softmax operation.

            Given the output of a forward softmax operation and the gradient of the loss with respect to that output,
            this function computes the gradient with respect to the original input of the softmax operation.

            The backward pass for softmax is defined as:

            .. math::
                \frac{\partial L}{\partial x_i} = y_i \cdot \left(\frac{\partial L}{\partial y_i} - \sum_{j=1}^{K} y_j \cdot \frac{\partial L}{\partial y_j}\right)

            where :math:`y_i` is the softmax output and :math:`\frac{\partial L}{\partial y_i}` is the incoming gradient.

            Args:
                softmax_output_tensor (ttnn.Tensor): The output tensor from the forward softmax operation.
                grad_tensor (ttnn.Tensor): The gradient tensor with respect to the softmax output.
                dim (int, optional): The dimension along which softmax was computed in the forward pass. Defaults to -1 (last dimension).

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. If not provided, inherits from input tensor.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.

            Returns:
                ttnn.Tensor: Gradient tensor with respect to the original softmax input, with the same shape as the input tensors.

            Note:
                Both input tensors must have the same shape and be the result of a forward softmax operation and its corresponding gradient.

            Supported dtypes and layouts

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            Example:
                .. code-block:: python

                    # Forward pass
                    input_tensor = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    softmax_output = ttnn.softmax(input_tensor, dim=-1)

                    # Assume we have gradients from subsequent operations
                    grad_output = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

                    # Backward pass
                    grad_input = ttnn.softmax_backward(softmax_output, grad_output, dim=-1)
    )doc";

    using OperationType = decltype(ttnn::softmax_backward);

    ttnn::bind_registered_operation(
        module,
        ttnn::softmax_backward,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& softmax_output_tensor,
               const ttnn::Tensor& grad_tensor,
               const uint32_t dim) -> ttnn::Tensor {
                return self(
                    softmax_output_tensor, grad_tensor, dim /*, memory_config, compute_kernel_config, numeric_stable*/);
            },
            py::arg("softmax_output_tensor").noconvert(),
            py::arg("grad_tensor").noconvert(),
            py::arg("dim") = -1});
}

void bind_normalization_softmax_backward(py::module& module) { bind_normalization_softmax_backward_operation(module); }
}  // namespace ttnn::operations::normalization::detail
