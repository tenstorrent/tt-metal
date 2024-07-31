// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace complex_unary {

namespace detail {

template <typename complex_unary_operation_t>
void bind_complex_unary_tensor(py::module& module, const complex_unary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
R"doc({0}(input_tensor: ComplexTensor, *, memory_config: ttnn.MemoryConfig) -> Tensor

{2}

Args:
    * :attr:`input_tensor` (ComplexTensor)

Keyword args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

Example:

    >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    >>> output = {1}(tensor)
)doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const complex_unary_operation_t& self,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> Tensor {
                return self(input_tensor, memory_config);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename complex_unary_operation_t>
void bind_complex_unary_complextensor(py::module& module, const complex_unary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
R"doc({0}(input_tensor: ComplexTensor, *, memory_config: ttnn.MemoryConfig) -> ComplexTensor

{2}

Args:
    * :attr:`input_tensor` (ComplexTensor)

Keyword args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): memory config for the output tensor

Example:

    >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device)
    >>> output = {1}(tensor)
)doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const complex_unary_operation_t& self,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> ComplexTensor {
                return self(input_tensor, memory_config);
            },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_complex_unary_tensor(
        module,
        ttnn::real,
        R"doc(Performs complex operations for real of :attr:`input_tensor`.)doc");

    detail::bind_complex_unary_tensor(
        module,
        ttnn::imag,
        R"doc(Performs complex operations for imag of :attr:`input_tensor`.)doc");

    detail::bind_complex_unary_tensor(
        module,
        ttnn::angle,
        R"doc(Performs complex operations for angle of :attr:`input_tensor`.)doc");

    detail::bind_complex_unary_tensor(
        module,
        ttnn::is_imag,
        R"doc(Returns boolean tensor if value of :attr:`input_tensor` is imag.)doc");

    detail::bind_complex_unary_tensor(
        module,
        ttnn::is_real,
        R"doc(Returns boolean tensor if value of :attr:`input_tensor` is real.)doc");

    detail::bind_complex_unary_complextensor(
        module,
        ttnn::conj,
        R"doc(Returns complex conjugate value of complex tensor :attr:`input_tensor`.)doc");

    detail::bind_complex_unary_complextensor(
        module,
        ttnn::polar,
        R"doc(Perform an polar to Cartesian transformation on :attr:`input_tensor`, input_tensor.real(r), input_tensor.imag(theta) into x + i*y generating a complex tensor.)doc");

}

}  // namespace complex_unary
}  // namespace operations
}  // namespace ttnn
