// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "downsample.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace py = pybind11;

namespace ttnn::operations::downsample {
namespace detail {

void bind_downsample(py::module& module) {
    const auto doc = R"doc(
 Downsamples a given multi-channel 2D (spatial) data.
 The input data is assumed to be of the form [N, H, W, C].

 Args:
     * :attr:`input_tensor`: the input tensor
     * :attr:`downsample_params`: Params list: batch size, conv input H, conv input W, conv stride H, conv stride W
     )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::downsample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("downsample_params"), py::arg("dtype") = std::nullopt});
}

}  // namespace detail
void py_module(py::module& module) { detail::bind_downsample(module); }

}  // namespace ttnn::operations::downsample
