// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "upsample.hpp"
#include "upsample3d.hpp"

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
            scale_factor (int or tt::tt_metal::Array2D): multiplier for spatial size.


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::upsample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("scale_factor"),
            py::kw_only(),
            py::arg("mode") = "nearest",
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

void bind_upsample3d(py::module& module) {
    const auto doc = R"doc(
        Upsamples a given multi-channel 3D (volumetric) data.
        The input data is assumed to be of the form [N, D, H, W, C].

        The algorithm available for upsampling is 'nearest' neighbor.

        Args:
            input_tensor (ttnn.Tensor): the input tensor with shape [N, D, H, W, C].
            scale_factor (int or tuple of 3 ints): multiplier for spatial size. If int, same factor is used for all dimensions.

        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with shape [N, D*scale_d, H*scale_h, W*scale_w, C].

        Example:
            >>> input = torch.ones((1, 2, 4, 4, 16), dtype=torch.bfloat16)
            >>> tt_input = ttnn.from_torch(input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
            >>> output = ttnn.upsample3d(tt_input, scale_factor=2)
            >>> # Output shape: (1, 4, 8, 8, 16)
        )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::upsample3d,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("scale_factor"), py::kw_only(), py::arg("memory_config") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_upsample(module);
    detail::bind_upsample3d(module);
}

}  // namespace ttnn::operations::upsample
