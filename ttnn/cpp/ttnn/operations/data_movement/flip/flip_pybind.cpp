// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flip_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/flip/flip.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_flip(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self, const ttnn::Tensor& input_tensor, const std::vector<int>& dims)
                -> ttnn::Tensor { return self(input_tensor, dims); },
            py::arg("input_tensor"),
            py::arg("dims")});
}

}  // namespace detail

void py_bind_flip(pybind11::module& module) {
    detail::bind_flip(
        module,
        ttnn::flip,
        R"doc(flip(input_tensor: ttnn.Tensor, dims: List[int]) -> ttnn.Tensor


       Flips a tensor along multiple specified dimensions.


       Args:
           * :attr:`input_tensor`: The tensor to flip.
           * :attr:`dims`: List of dimensions along which to flip.


       )doc");
}

}  // namespace ttnn::operations::data_movement
