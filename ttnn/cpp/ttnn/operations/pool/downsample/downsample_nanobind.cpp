// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "downsample_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "downsample.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::downsample {
namespace {

void bind_downsample(nb::module_& mod, const char* doc) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::downsample,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"), nb::arg("downsample_params"), nb::arg("dtype") = std::nullopt});
}

}  // namespace
void bind_downsample(nb::module_& mod) {
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
    bind_downsample(mod, doc);
}

}  // namespace ttnn::operations::downsample
