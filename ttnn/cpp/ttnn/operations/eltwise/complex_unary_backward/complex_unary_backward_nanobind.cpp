// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_backward_nanobind.hpp"

#include <string>
#include <string_view>

#include <fmt/format.h>
#include <nanobind/nanobind.h>

#include "cpp/ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/eltwise/complex_unary_backward/complex_unary_backward.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::complex_unary_backward {

namespace {

template <typename complex_unary_backward_operation_t>
void bind_complex_unary_backward(
    nb::module_& mod,
    const complex_unary_backward_operation_t& operation,
    const std::string& description,
    const std::string_view supported_dtype = "") {
    auto doc = fmt::format(
        R"doc(
        {2}


        Args:
            grad_tensor (ComplexTensor): the input tensor.
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation.


        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            {3}

        Example:
            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, tensor)


        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const complex_unary_backward_operation_t& self,
               const ComplexTensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> std::vector<ComplexTensor> {
                return self(grad_tensor, input_tensor, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

template <typename complex_unary_backward_operation_t>
void bind_complex_unary_backward_tensor(
    nb::module_& mod,
    const complex_unary_backward_operation_t& operation,
    const std::string& description,
    const std::string_view supported_dtype = "") {
    auto doc = fmt::format(
        R"doc(
        {2}


        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation.


        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            {3}

        Example:
            >>> grad_tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> tensor = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
            >>> output = {1}(grad_tensor, tensor)


        )doc",

        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const complex_unary_backward_operation_t& self,
               const ttnn::Tensor& grad_tensor,
               const ComplexTensor& input_tensor,
               const ttnn::MemoryConfig& memory_config) -> std::vector<ComplexTensor> {
                return self(grad_tensor, input_tensor, memory_config);
            },
            nb::arg("grad_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config")});
}

}  // namespace

void py_module(nb::module_& mod) {
    bind_complex_unary_backward(
        mod,
        ttnn::polar_bw,
        R"doc(Performs backward operations for complex polar function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(Supported dtypes, layouts, and ranks:

           +----------------------------+---------------------------------+-------------------+
           |     Dtypes                 |         Layouts                 |     Ranks         |
           +----------------------------+---------------------------------+-------------------+
           |    BFLOAT16                |       TILE                      |      2, 3, 4      |
           +----------------------------+---------------------------------+-------------------+

        )doc");

    bind_complex_unary_backward(
        mod,
        ttnn::conj_bw,
        R"doc(Performs backward operations for complex conj function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_complex_unary_backward_tensor(
        mod,
        ttnn::imag_bw,
        R"doc(Performs backward operations for complex imaginary function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_complex_unary_backward_tensor(
        mod,
        ttnn::real_bw,
        R"doc(Performs backward operations for complex real function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc");

    bind_complex_unary_backward_tensor(
        mod,
        ttnn::angle_bw,
        R"doc(Performs backward operations for complex angle function on :attr:`input_tensor` with given :attr:`grad_tensor`.)doc",
        R"doc(Supported dtypes, layouts, and ranks:

           +----------------------------+---------------------------------+-------------------+
           |     Dtypes                 |         Layouts                 |     Ranks         |
           +----------------------------+---------------------------------+-------------------+
           |    BFLOAT16                |       TILE                      |      2, 3, 4      |
           +----------------------------+---------------------------------+-------------------+

        )doc");
}

}  // namespace ttnn::operations::complex_unary_backward
