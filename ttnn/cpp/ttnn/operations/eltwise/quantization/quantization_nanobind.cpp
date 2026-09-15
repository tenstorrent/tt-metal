// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization_nanobind.hpp"

#include <optional>
#include <string>
#include <variant>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "quantization.hpp"

namespace ttnn::operations::quantization {
namespace {

void bind_quantize(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Quantizes a floating-point tensor into an integer tensor: ``q = input_tensor / scale + zero_point``,
        per tensor by default or per channel along :attr:`axis`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale (ttnn.Tensor or Number): the quantization scale.
            zero_point (ttnn.Tensor or Number): the quantization zero point.

        Keyword Args:
            axis (int, optional): the axis of the quantization dimension of the input tensor. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {0}
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

            When :attr:`scale` and :attr:`zero_point` are tensors, they must be FLOAT32.
        )doc",
        "BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32");

    ttnn::bind_function<"quantize">(
        mod,
        doc.c_str(),
        &ttnn::quantize,
        nb::arg("input_tensor"),
        nb::arg("scale"),
        nb::arg("zero_point"),
        nb::kw_only(),
        nb::arg("axis") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

void bind_requantize(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Re-expresses an already quantized tensor on a different scale and zero point:
        ``q' = (input_tensor - in_zero_point) * in_scale / out_scale + out_zero_point``.
        Equivalent to dequantizing and then quantizing with the output scale and zero point.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            in_scale (ttnn.Tensor or Number): the input quantization scale.
            in_zero_point (ttnn.Tensor or Number): the input quantization zero point.
            out_scale (ttnn.Tensor or Number): the output quantization scale.
            out_zero_point (ttnn.Tensor or Number): the output quantization zero point.

        Keyword Args:
            axis (int, optional): the axis of the quantization dimension of the input tensor. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {0}
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

            Input tensor dtype must be INT32. When scale and zero-point parameters are tensors, they must be FLOAT32.

            **Mixed Quantization Support:**

            This operation supports mixed quantization schemes:

            - **Per-tensor to Per-channel**: Convert from global quantization parameters to per-channel parameters along the specified axis.
            - **Per-channel to Per-tensor**: Convert from per-channel quantization parameters to global parameters.
            - **Per-tensor to Per-tensor**: Standard requantization with scalar parameters.
            - **Per-channel to Per-channel**: Requantization with per-channel parameters along the same axis.

            **Execution Paths:**

            When all four parameters (in_scale, in_zero_point, out_scale, out_zero_point) are provided as tensors and an axis is specified:

            - The operation uses a path with explicit shape expansion and broadcasting.
            - Per-tensor parameters (scalar tensors) are broadcast to match the input tensor shape.
            - Per-channel parameters (1D tensors) are reshaped and expanded along the specified axis.
            - The implementation performs the mathematical requantization in floating point and typecasts to the output dtype: q' = q * (s_in/s_out) + (z_out - z_in * s_in/s_out).

            When all four parameters are provided as scalar values (float/int32):

            - Uses a path with a specialized kernel operation.
            - Computes the requantization directly in a single fused operation.

            When there is a mix of scalar and tensor parameters:

            - Falls back to a composite operation path.
            - Decomposes requantization into separate dequantize and quantize operations.
        )doc",
        "INT32");

    ttnn::bind_function<"requantize">(
        mod,
        doc.c_str(),
        &ttnn::requantize,
        nb::arg("input_tensor"),
        nb::arg("in_scale"),
        nb::arg("in_zero_point"),
        nb::arg("out_scale"),
        nb::arg("out_zero_point"),
        nb::kw_only(),
        nb::arg("axis") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

void bind_dequantize(nb::module_& mod) {
    auto doc = fmt::format(
        R"doc(
        Converts a quantized integer tensor back to floating point: ``t = (input_tensor - zero_point) * scale``,
        per tensor by default or per channel along :attr:`axis`.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale (ttnn.Tensor or Number): the quantization scale.
            zero_point (ttnn.Tensor or Number): the quantization zero point.

        Keyword Args:
            axis (int, optional): the axis of the quantization dimension of the input tensor. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {0}
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

            Input tensor dtype must be INT32. When :attr:`scale` and :attr:`zero_point` are tensors, they must be FLOAT32.
        )doc",
        "INT32");

    ttnn::bind_function<"dequantize">(
        mod,
        doc.c_str(),
        &ttnn::dequantize,
        nb::arg("input_tensor"),
        nb::arg("scale"),
        nb::arg("zero_point"),
        nb::kw_only(),
        nb::arg("axis") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace

void py_module(nb::module_& mod) {
    bind_quantize(mod);
    bind_requantize(mod);
    bind_dequantize(mod);
}
}  // namespace ttnn::operations::quantization
