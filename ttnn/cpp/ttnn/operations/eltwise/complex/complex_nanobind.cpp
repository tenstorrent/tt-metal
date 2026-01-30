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
                real (ttnn.Tensor): the real part tensor.
                imag (ttnn.Tensor): the imaginary part tensor.

            Returns:
                ttnn.ComplexTensor: the complex tensor with real and imag parts.

            Note:
                This operation supports tensors according to the following data types and layouts:

                .. list-table:: real and imag tensors
                    :header-rows: 1

                    * - dtype
                        - layout
                    * - BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, UINT32, INT32, UINT16, UINT8
                        - TILE
                    * - BFLOAT16, FLOAT32, UINT32, INT32, UINT16, UINT8
                        - ROW_MAJOR

                Memory Support:
                    - Interleaved: DRAM and L1
                    - Height, Width, Block, and ND Sharded: DRAM and L1

                Limitations:
                    -  The real and imag tensors must have the same shape and layout (both must be TILE or ROW_MAJOR).
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
