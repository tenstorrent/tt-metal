// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
        {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale (ttnn.Tensor or Number): the quantization scale.
            zero_point (ttnn.Tensor or Number): the quantization zero point.

        Keyword Args:
            axis (Number, optional): the axis of the quantization dimension of the input tensor.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> input_tensor = ttnn.from_torch(torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scale = 0.001173
            >>> zero_point = -213
            >>> output = {1}(input_tensor, scale, zero_point)
        )doc",
        "quantize",
        "ttnn.quantize",
        "Quantize Operation",
        "BFLOAT16");

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
        {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            in_scale (ttnn.Tensor or Number): the input quantization scale.
            in_zero_point (ttnn.Tensor or Number): the input quantization zero point.
            out_scale (ttnn.Tensor or Number): the output quantization scale.
            out_zero_point (ttnn.Tensor or Number): the output quantization zero point.

        Keyword Args:
            axis (Number, optional): the axis of the quantization dimension of the input tensor.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

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

        Example:
            >>> input_tensor = ttnn.from_torch(torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> in_scale = 0.001173
            >>> in_zero_point = -213
            >>> out_scale = 0.002727
            >>> out_zero_point = -73
            >>> output = {1}(input_tensor, in_scale, in_zero_point, out_scale, out_zero_point)
        )doc",
        "requantize",
        "ttnn.requantize",
        "Re-quantize Operation",
        "BFLOAT16");

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
        {2}

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            scale (ttnn.Tensor or Number): the quantization scale.
            zero_point (ttnn.Tensor or Number): the quantization zero point.

        Keyword Args:
            axis (Number, optional): the axis of the quantization dimension of the input tensor.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - {3}
                 - TILE

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> input_tensor = ttnn.from_torch(torch.tensor([[-127 -42], [43 127]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scale = 0.001173
            >>> zero_point = -213
            >>> output = {1}(input_tensor, scale, zero_point)
        )doc",
        "dequantize",
        "ttnn.dequantize",
        "De-quantize Operation",
        "BFLOAT16");

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
