// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_view(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const ttnn::Shape& shape)
                -> ttnn::Tensor { return self(input_tensor, shape); },
            py::arg("input_tensor"),
            py::arg("shape"),
        },
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int32_t> shape) -> ttnn::Tensor { return self(input_tensor, shape); },
            py::arg("input_tensor"),
            py::arg("shape"),
        });
}

}  // namespace detail

void py_bind_view(pybind11::module& module) {
    detail::bind_view(
        module,
        ttnn::view,
        R"doc(

        This is a 0 cost view operation that returns the same tensor that was passed to it but with a new shape

        Note: The following conditions must be met:
            * the memory is stored on the device
            * the last dimension must not change
            * In Tiled the second last two dimensions must not change OR there is no padding on the second last dimension

        Args:
            * input_tensor: Input Tensor.
            * new_shape: New shape of tensor.

        Returns:
            ttnn.Tensor: a reference to the input tensor but with the new shape.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((2, 1, 1, 1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.view(tensor, (2, 1, 4))

        )doc");

}  // namespace ttnn::operations::data_movement
}  // namespace ttnn::operations::data_movement
