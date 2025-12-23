// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deform_conv2d_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"

#include "ttnn/types.hpp"
#include <tt-metalium/constants.hpp>

#include "ttnn/operations/conv/deform_conv2d/deform_conv2d.hpp"

namespace ttnn::operations::conv::deform_conv2d {

void bind_deform_conv2d(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::deform_conv2d,
        R"doc(
        Applies a 2D **deformable convolution** over an input tensor, allowing the sampling
        locations in the standard convolution kernel to shift dynamically using learned offsets.

        This operation is an extension of standard convolution where:
        • The input is sampled at positions offset by a learned offset field.
        • Offsets allow the kernel to model geometric variations more effectively.

        ---------------------------------------------------------------------
        Parameters
        ---------------------------------------------------------------------

        :param ttnn.Tensor input_tensor:
            The input feature map in **NHWC** format `[B, H, W, C_in]`.
            The tensor may reside on host or device.

        :param ttnn.Tensor weight_tensor:
            The convolution kernel weights.
            Expected format:
                `[kH, kW, C_in // groups, C_out]`

            • `kH`, `kW` → kernel spatial size
            • Channels must be divisible by `groups`.

        :param ttnn.Tensor offset_tensor:
            The learned 2D offsets for deformable sampling.
            Expected shape:
                `[B, H_out, W_out, 2 * kH * kW * offset_groups]`

            Each kernel position has a pair of offsets `(dy, dx)`.
            If `offset_groups > 1`, the offsets are split among groups.

        :param int stride:
            Convolution stride. Default: 1.

        :param int padding:
            Zero-padding on all sides of the input. Default: 0.

        :param int dilation:
            Spacing between kernel elements. Default: 1.

        :param int groups:
            The number of groups for grouped convolution.
            • Input channels are split into `groups` partitions.
            • Weight tensor’s `C_in // groups` must match per-group input.

        :param int offset_groups:
            Number of groups for the offset tensor.
            • Offsets are partitioned into `offset_groups`.
            • Must satisfy: `offset_groups <= groups`.

        :param ttnn.DataType, None dtype:
            Output tensor datatype. Default: infer from `input_tensor`.

        :param ttnn.MemoryConfig, None memory_config:
            Memory configuration for the output tensor.

        ---------------------------------------------------------------------
        Returns
        ---------------------------------------------------------------------

        :return:
            Output tensor of the deformable convolution.

        :rtype:
            ttnn.Tensor in **NHWC** format `[B, H_out, W_out, C_out]`.

        ---------------------------------------------------------------------
        Example
        ---------------------------------------------------------------------

            ttnn.deform_conv2d_pre(
                tt_input,
                tt_weight,
                tt_offset,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                offset_groups=offset_groups
            )

        )doc",

        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::deform_conv2d)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weight_tensor,
               const ttnn::Tensor& offset_tensor,
               const int stride,
               const uint32_t padding,
               const int dilation,
               const int groups,
               const int offset_groups) -> ttnn::Tensor {
                return self(
                    input_tensor, weight_tensor, offset_tensor, stride, padding, dilation, groups, offset_groups);
            },
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("offset_tensor"),
            nb::arg("stride"),
            nb::arg("padding"),
            nb::arg("dilation"),
            nb::arg("groups"),
            nb::arg("offset_groups")});
}

}  // namespace ttnn::operations::conv::deform_conv2d
