// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_pybind.hpp"
#include "reshape.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/types.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"


namespace ttnn::operations::experimental::reshape::detail {
namespace py = pybind11;

void py_bind_reshape(py::module& module) {
    auto doc = R"doc(

        Note: for a 0 cost view, the following conditions must be met:
            * the last dimension must not change
            * In Tiled the second last two dimensions must not change OR there is no padding on the second last dimension

        Args:
            * input_tensor: Input Tensor.
            * new_shape: New shape of tensor.

        Returns:
            ttnn.Tensor: the output tensor with the new shape.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.experimental.reshape(tensor, (1, 1, 2, 2))

        )doc";
    bind_registered_operation(
        module,
        ttnn::experimental::reshape,
        doc,
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::reshape)& self, ttnn::Tensor& input_tensor, int N, int C, int H, int W) {
                return self(input_tensor, infer_dims_for_reshape(input_tensor, ttnn::SmallVector<int>{N, C, H, W}));
            },
            py::arg("input_tensor"),
            py::arg("N"),
            py::arg("C"),
            py::arg("H"),
            py::arg("W"),
        },

        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::reshape)& self, ttnn::Tensor& input_tensor, const ttnn::Shape& shape) {
                return self(input_tensor, shape); },
            py::arg("input_tensor"),
            py::arg("shape"),
        },
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::experimental::reshape)& self, ttnn::Tensor& input_tensor, const ttnn::SmallVector<int32_t>& shape) {
                return self(input_tensor, infer_dims_for_reshape(input_tensor, shape));
            },
            py::arg("input_tensor"),
            py::arg("shape"),
        });
}

}  // namespace ttnn::operations::experimental::reshape::detail
