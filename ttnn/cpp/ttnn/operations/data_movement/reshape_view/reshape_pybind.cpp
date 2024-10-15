// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_reshape_view(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SimpleShape& shape
               ) -> ttnn::Tensor {
                return self(input_tensor, shape);
            },
            py::arg("input_tensor"),
            py::arg("shape"),
            },
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::vector<int32_t>& shape
               ) -> ttnn::Tensor {
                return self(input_tensor, shape);
            },
            py::arg("input_tensor"),
            py::arg("shape"),
            }
        );
}

}  // namespace detail


void py_bind_reshape_view(pybind11::module& module) {
    detail::bind_reshape_view(
        module,
        ttnn::reshape,

        R"doc(

        Args:
            * input_tensor: Input Tensor.
            * new_shape: New shape of tensor.

        Returns:
            ttnn.Tensor: the output tensor with the new shape.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.reshape(tensor, (1, 1, 2, 2))

        )doc");
}

}  // namespace ttnn::operations::data_movement
