// // SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

#include "roll_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
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
               const ttnn::SmallVector<int>& shifts,
               const ttnn::SmallVector<int>& dim) -> ttnn::Tensor { return self(input_tensor, shifts, dim); },
            py::arg("input_tensor"),
            py::arg("shifts"),
            py::arg("dim")},

        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const int shifts, const int dim)
                -> ttnn::Tensor { return self(input_tensor, shifts, dim); },
            py::arg("input_tensor"),
            py::arg("shifts"),
            py::arg("dim")},

        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const int shifts)
                -> ttnn::Tensor { return self(input_tensor, shifts); },
            py::arg("input_tensor"),
            py::arg("shifts")});
}

}  // namespace detail

void py_bind_roll(pybind11::module& module) {
    detail::bind_roll(
        module,
        ttnn::roll,
        R"doc(
        roll(input_tensor: ttnn.Tensor, shifts: Union[int, List[int]], dim: Union[int, List[int]]) -> ttnn.Tensor

        Performs circular shifting of elements along the specified dimension(s).

        Args:
            input_tensor: A tensor whose elements will be rolled.
            shifts: The number of places by which elements are shifted. Can be an integer or a list of integers (one per dimension).
            dim: The dimension(s) along which to roll. If shifts is a list, then dim must be a list of the same length as shifts.
        )doc");
}

}  // namespace ttnn::operations::data_movement
