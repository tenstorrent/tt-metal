// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"

#include "upsample.hpp"

namespace ttnn::operations::upsample {
namespace {

void bind_upsample(nb::module_& mod) {
    const auto* const doc = R"doc(
        Upsamples a given multi-channel 2D (spatial) data.
        The input data is assumed to be of the form [N, H, W, C].

        Supports both integer and floating-point scale factors with automatic routing
        to optimized implementations:
        - Integer scales: Optimized path with reader/writer replication (fastest)
        - Float scales: General path with coordinate mapping (supports fractional scales)

        The algorithms available for upsampling are 'nearest' and 'bilinear'.
        Note: 'bilinear' mode requires integer scale factors.


        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale_factor (int, float, [int, int], or [float, float]): multiplier for spatial size.
                - int: uniform integer scale for both H and W
                - float: uniform float scale for both H and W
                - [int, int]: separate integer scales for [H, W]
                - [float, float]: separate float scales for [H, W]


        Keyword args:
            mode (str, optional): upsampling mode - 'nearest' or 'bilinear'. Defaults to 'nearest'.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.


        Examples:
            >>> # Integer scale (optimized path)
            >>> output = ttnn.upsample(input, 2)
            >>> output = ttnn.upsample(input, [2, 3])

            >>> # Float scale (general path)
            >>> output = ttnn.upsample(input, 1.5)
            >>> output = ttnn.upsample(input, [1.5, 2.5])

            >>> # Bilinear mode (integer scales only)
            >>> output = ttnn.upsample(input, 2, mode="bilinear")

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
