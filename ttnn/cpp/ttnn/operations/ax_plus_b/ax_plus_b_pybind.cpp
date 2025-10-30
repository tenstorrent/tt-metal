// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ax_plus_b_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/ax_plus_b/device/ax_plus_b_device_operation.hpp"

namespace ttnn::operations::ax_plus_b {

void bind_ax_plus_b_operation(py::module& module) {
    auto doc = R"doc(
        Performs the ax + b operation where a and b are tensors and x is the input tensor.

        .. math::
            \mathrm{output\_tensor} = \mathrm{tensor\_a} \times \mathrm{tensor\_x} + \mathrm{tensor\_b}

        Args:
            tensor_a (ttnn.Tensor): the 'a' coefficient tensor.
            tensor_x (ttnn.Tensor): the input tensor 'x'.
            tensor_b (ttnn.Tensor): the 'b' bias tensor.

        Keyword args:
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            This operation computes ax + b where:
            - a is the coefficient tensor (tensor_a)
            - x is the input tensor (tensor_x)
            - b is the bias tensor (tensor_b)

        Example:
            >>> tensor_a = ttnn.from_torch(torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor_x = ttnn.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> tensor_b = ttnn.from_torch(torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.prim.ax_plus_b(tensor_a, tensor_x, tensor_b)
        )doc";

    bind_registered_operation(
        module,
        ttnn::prim::ax_plus_b,
        doc,
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::prim::ax_plus_b)& self,
               const ttnn::Tensor& tensor_a,
               const ttnn::Tensor& tensor_x,
               const ttnn::Tensor& tensor_b,
               const std::optional<ttnn::Tensor>& tensor_y) -> ttnn::Tensor {
                return self(tensor_a, tensor_x, tensor_b, tensor_y);
            },
            py::arg("tensor_a"),
            py::arg("tensor_x"),
            py::arg("tensor_b"),
            py::arg("output_tensor") = std::nullopt});
}

void py_module(py::module& module) { bind_ax_plus_b_operation(module); }

}  // namespace ttnn::operations::ax_plus_b
