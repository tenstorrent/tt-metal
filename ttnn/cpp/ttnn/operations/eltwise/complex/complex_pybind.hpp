// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "complex.hpp"

namespace py = pybind11;

namespace ttnn::operations::complex {

namespace detail {

void bind_complex_tensor_type(py::module& m) {
    py::class_<ComplexTensor>(m, "ComplexTensor")
        .def(py::init<std::array<Tensor, 2>>())
        .def_property_readonly("real", &ComplexTensor::real)
        .def_property_readonly("imag", &ComplexTensor::imag)
        .def("deallocate", &ComplexTensor::deallocate)
        .def("__getitem__", &ComplexTensor::operator[]);
}

void bind_complex_tensor(py::module& module) {
    auto doc = fmt::format(
        R"doc({0}real: ttnn.Tensor, imag: ttnn.Tensor -> ComplexTensor

            Create a complex tensor from real and imaginary part tensors.

            Args:
                * :attr:`real`
                * :attr:`imag`

            Example:

                >>> real = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> imag = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device)
                >>> complex_tensor = ttnn.complex_tensor(real, imag)
        )doc",
        ttnn::complex_tensor.base_name());

    bind_registered_operation(
        module, ttnn::complex_tensor, doc, ttnn::pybind_arguments_t{py::arg("real"), py::arg("imag")});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_complex_tensor_type(module);
    detail::bind_complex_tensor(module);
}

}  // namespace ttnn::operations::complex
