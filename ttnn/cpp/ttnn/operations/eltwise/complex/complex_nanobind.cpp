// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_nanobind.hpp"

#include <tuple>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/decorators.hpp"

#include "complex.hpp"

namespace ttnn::operations::complex {

namespace {

void bind_complex_tensor_type(nb::module_& mod) {
    nb::class_<ComplexTensor>(mod, "ComplexTensor")
        .def(nb::init<std::tuple<const Tensor&, const Tensor&>>())
        .def_prop_ro("real", &ComplexTensor::real)
        .def_prop_ro("imag", &ComplexTensor::imag)
        .def("deallocate", &ComplexTensor::deallocate)
        .def("__getitem__", &ComplexTensor::operator[]);
}

void bind_complex_tensor(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
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
        mod, ttnn::complex_tensor, doc, ttnn::nanobind_arguments_t{nb::arg("real"), nb::arg("imag")});
}

}  // namespace

void py_module(nb::module_& mod) {
    bind_complex_tensor_type(mod);
    bind_complex_tensor(mod);
}

}  // namespace ttnn::operations::complex
