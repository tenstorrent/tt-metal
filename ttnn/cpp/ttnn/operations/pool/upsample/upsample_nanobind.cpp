// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "upsample.hpp"

namespace ttnn::operations::upsample {
namespace {

void bind_upsample(nb::module_& mod) {
    const auto doc = R"doc(
        Upsamples a given multi-channel 2D (spatial) data.
        The input data is assumed to be of the form [N, H, W, C].

        The algorithms available for upsampling are 'nearest' for now.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale_factor (int or tt::tt_metal::Array2D): multiplier for spatial size.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::upsample,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("scale_factor"),
            nb::kw_only(),
            nb::arg("mode") = "nearest",
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none()});
}

}  // namespace

void py_module(nb::module_& mod) { bind_upsample(mod); }
}  // namespace ttnn::operations::upsample
