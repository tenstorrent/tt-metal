// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "madd_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"

#include "madd.hpp"

namespace ttnn::operations::madd {
namespace {

void bind_madd(nb::module_& mod) {
    const auto* const doc = R"doc(
        Performs elementwise multiply-add operation on given tensors.
        Computes: output = a * b + c


        Args:
            a (ttnn.Tensor): the first input tensor.
            b (ttnn.Tensor): the second input tensor.
            c (ttnn.Tensor): the third input tensor.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        Examples:
            >>> output = ttnn.madd(input_a, input_b, input_c)
        )doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::madd,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("c"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace

void py_module(nb::module_& mod) { bind_madd(mod); }
}  // namespace ttnn::operations::madd
