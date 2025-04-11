// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"

#include "gridsample.hpp"

namespace ttnn::operations::gridsample {

namespace detail {

namespace py = pybind11;

void bind_gridsample(py::module& module) {
    const auto doc = R"doc(
        gridsample a given multi-channel 2D (spatial) data.
        The input data is assumed to be of the form [N, H, W, C].

        The algorithms available for gridsample are 'nearest' for now.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            grid (ttnn.Tensor): the grid tensor.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc";

    using OperationType = decltype(ttnn::gridsample);
    ttnn::bind_registered_operation(
        module,
        ttnn::gridsample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("grid"),
            py::kw_only(),
            py::arg("mode") = "bilinear",
            py::arg("align_corners") = "false",
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace detail
void py_module(py::module& module) { detail::bind_gridsample(module); }
}  // namespace ttnn::operations::gridsample
