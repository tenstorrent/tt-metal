// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "upsample.hpp"

namespace ttnn::operations::upsample {

namespace detail {

namespace py = pybind11;

void bind_upsample(py::module& module) {
    const auto doc = R"doc(
 Upsamples a given multi-channel 2D (spatial) data.
 The input data is assumed to be of the form [N, H, W, C].

 The algorithms available for upsampling are 'nearest' for now.

 Args:
     * :attr:`input_tensor`: the input tensor
     * :attr:`scale_factor`: multiplier for spatial size. Has to match input size if it is a tuple.
     )doc";

    using OperationType = decltype(ttnn::upsample);
    ttnn::bind_registered_operation(
        module,
        ttnn::upsample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("scale_factor"), py::kw_only(), py::arg("memory_config") = std::nullopt});
}


} //detail
void py_module(py::module& module) { detail::bind_upsample(module); }
} //namespace ttnn::operations::upsample
