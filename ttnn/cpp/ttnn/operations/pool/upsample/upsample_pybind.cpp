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
            input_tensor (ttnn.Tensor): the input tensor.
            scale_factor (int or tt::tt_metal::Array2D or tt::tt_metal::Array3D or tt::tt_metal::Array4D): multiplier for spatial size. Has to match input size if it is a tuple.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc";

    using OperationType = decltype(ttnn::upsample);
    ttnn::bind_registered_operation(module,
                                    ttnn::upsample,
                                    doc,
                                    ttnn::pybind_arguments_t{py::arg("input_tensor"),
                                                             py::arg("scale_factor"),
                                                             py::kw_only(),
                                                             py::arg("mode") = "nearest",
                                                             py::arg("memory_config") = std::nullopt,
                                                             py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace detail
void py_module(py::module& module) {
    detail::bind_upsample(module);
}
}  // namespace ttnn::operations::upsample
