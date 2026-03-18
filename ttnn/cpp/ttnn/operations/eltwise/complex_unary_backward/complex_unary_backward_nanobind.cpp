// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_backward_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/eltwise/complex_unary_backward/complex_unary_backward.hpp"

namespace ttnn::operations::complex_unary_backward {

void py_module(nb::module_& mod) {
    ttnn::bind_function<"polar_bw">(
        mod,
        R"doc(
        Performs backward operations for complex polar function on :attr:`input_tensor` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ComplexTensor): the input tensor.
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation.


        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32
                 - TILE, ROW_MAJOR

        )doc",
        &ttnn::polar_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config"));

    ttnn::bind_function<"conj_bw">(
        mod,
        R"doc(
        Performs backward operations for complex conj function on :attr:`input_tensor` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ComplexTensor): the input tensor.
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation.


        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:

        )doc",
        &ttnn::conj_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config"));

    ttnn::bind_function<"imag_bw">(
        mod,
        R"doc(
        Performs backward operations for complex imaginary function on :attr:`input_tensor` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation.


        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:

        )doc",
        &ttnn::imag_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config"));

    ttnn::bind_function<"real_bw">(
        mod,
        R"doc(
        Performs backward operations for complex real function on :attr:`input_tensor` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation.


        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:

        )doc",
        &ttnn::real_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config"));

    ttnn::bind_function<"angle_bw">(
        mod,
        R"doc(
        Performs backward operations for complex angle function on :attr:`input_tensor` with given :attr:`grad_tensor`.


        Args:
            grad_tensor (ttnn.Tensor): the input tensor.
            input_tensor (ComplexTensor): the input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig): Memory configuration for the operation.


        Returns:
            List of ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32
                 - TILE, ROW_MAJOR

        )doc",
        &ttnn::angle_bw,
        nb::arg("grad_tensor"),
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("memory_config"));
}

}  // namespace ttnn::operations::complex_unary_backward
