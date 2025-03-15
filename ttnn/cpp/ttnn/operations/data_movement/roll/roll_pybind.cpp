// // SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

#include "roll_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/data_movement/roll/roll.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_roll(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const py::object& shifts,
               const py::object& dim) -> ttnn::Tensor {
                std::vector<int> shift_values;
                std::vector<int> dim_values;

                if (py::isinstance<py::int_>(shifts)) {
                    shift_values.push_back(shifts.cast<int>());
                } else if (py::isinstance<py::list>(shifts)) {
                    shift_values = shifts.cast<std::vector<int>>();
                } else {
                    throw std::invalid_argument("shifts must be an int or a list of ints.");
                }

                if (py::isinstance<py::int_>(dim)) {
                    dim_values.push_back(dim.cast<int>());
                } else if (py::isinstance<py::list>(dim)) {
                    dim_values = dim.cast<std::vector<int>>();
                } else {
                    throw std::invalid_argument("dim must be an int or a list of ints.");
                }

                return self(input_tensor, shift_values, dim_values);
            },
            py::arg("input_tensor"),
            py::arg("shifts"),
            py::arg("dim")});
}

}  // namespace detail

void py_bind_roll(pybind11::module& module) {
    detail::bind_roll(
        module,
        ttnn::roll,
        R"doc(roll(input_tensor: ttnn.Tensor, shifts: Union[int, List[int]], dim: Union[int, List[int]]) -> ttnn.Tensor
            Performs circular shifting of elements along the specified dimension(s).

            Args:
                * :attr:`input_tensor`: A tensor whose elements will be rolled.
                * :attr:`shifts`: The number of places by which elements are shifted. Can be an integer or a list of integers (one per dimension).
                * :attr:`dim`: The dimension(s) along which to roll. Must have the same length as `shifts` if a list is provided.
            )doc");
}

}  // namespace ttnn::operations::data_movement
