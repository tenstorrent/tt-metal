// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_pybind.hpp"

#include <string>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::complex_unary {

namespace {

template <typename complex_unary_operation_t>
void bind_complex_unary_tensor(
    py::module& module, const complex_unary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}


        Args:
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        Example:
            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
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
               const ttnn::MemoryConfig& memory_config) -> Tensor { return self(input_tensor, memory_config); },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

template <typename complex_unary_operation_t>
void bind_complex_unary_complextensor(
    py::module& module, const complex_unary_operation_t& operation, const std::string& description) {
    auto doc = fmt::format(
        R"doc(
        {2}


        Args:
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory config for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        Example:

            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device = device)
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
               const ttnn::MemoryConfig& memory_config) -> ComplexTensor { return self(input_tensor, memory_config); },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config")});
}

}  // namespace

void py_module(py::module& module) {
    bind_complex_unary_tensor(
        module, ttnn::real, R"doc(Performs complex operations for real of :attr:`input_tensor`.)doc");

    bind_complex_unary_tensor(
        module, ttnn::imag, R"doc(Performs complex operations for imag of :attr:`input_tensor`.)doc");

    bind_complex_unary_tensor(
        module, ttnn::angle, R"doc(Performs complex operations for angle of :attr:`input_tensor`.)doc");

    bind_complex_unary_tensor(
        module, ttnn::is_imag, R"doc(Returns boolean tensor if value of :attr:`input_tensor` is imag.)doc");

    bind_complex_unary_tensor(
        module, ttnn::is_real, R"doc(Returns boolean tensor if value of :attr:`input_tensor` is real.)doc");

    bind_complex_unary_complextensor(
        module, ttnn::conj, R"doc(Returns complex conjugate value of complex tensor :attr:`input_tensor`.)doc");

    bind_complex_unary_complextensor(
        module,
        ttnn::polar,
        R"doc(Perform an polar to Cartesian transformation on :attr:`input_tensor`, input_tensor.real(r), input_tensor.imag(theta) into x + i*y generating a complex tensor.)doc");
}

}  // namespace ttnn::operations::complex_unary
