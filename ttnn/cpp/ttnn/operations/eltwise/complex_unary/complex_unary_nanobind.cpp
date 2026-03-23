// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"

namespace ttnn::operations::complex_unary {

void py_module(nb::module_& mod) {
    ttnn::bind_function<"real">(
        mod,
        R"doc(
        Performs complex operations for real of :attr:`input_tensor`.

        Args:
            input_tensor (ComplexTensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        &ttnn::real,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"imag">(
        mod,
        R"doc(
        Performs complex operations for imag of :attr:`input_tensor`.

        Args:
            input_tensor (ComplexTensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        &ttnn::imag,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"angle">(
        mod,
        R"doc(
        Performs complex operations for angle of :attr:`input_tensor`.

        Args:
            input_tensor (ComplexTensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        &ttnn::angle,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"is_imag">(
        mod,
        R"doc(
        Returns boolean tensor if value of :attr:`input_tensor` is imag.

        Args:
            input_tensor (ComplexTensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        &ttnn::is_imag,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"is_real">(
        mod,
        R"doc(
        Returns boolean tensor if value of :attr:`input_tensor` is real.

        Args:
            input_tensor (ComplexTensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        &ttnn::is_real,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"conj">(
        mod,
        R"doc(
        Returns complex conjugate value of complex tensor :attr:`input_tensor`.

        Args:
            input_tensor (ComplexTensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory config for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        &ttnn::conj,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());

    ttnn::bind_function<"polar">(
        mod,
        R"doc(
        Perform an polar to Cartesian transformation on :attr:`input_tensor`, input_tensor.real(r), input_tensor.imag(theta) into x + i*y generating a complex tensor.

        Args:
            input_tensor (ComplexTensor): the input tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory config for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        )doc",
        &ttnn::polar,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::complex_unary
