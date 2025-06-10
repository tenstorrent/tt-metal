// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "downsample_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "downsample.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::downsample {
namespace {

void bind_downsample(py::module& module, const char* doc) {
    ttnn::bind_registered_operation(
        module,
        ttnn::downsample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("downsample_params"), py::arg("dtype") = std::nullopt});
}

}  // namespace
void py_bind_downsample(py::module& module) {
    const auto doc = R"doc(
        Downsamples a given multi-channel 2D (spatial) data.
        The input data is assumed to be of the form [N, H, W, C].


        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            downsample_params (List): Params list: batch size, conv input H, conv input W, conv stride H, conv stride W.


        Keyword Args:
            dtype (ttnn.DataType, optional): Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc";
    bind_downsample(module, doc);
}

}  // namespace ttnn::operations::downsample
