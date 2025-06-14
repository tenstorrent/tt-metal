// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization_nanobind.hpp"

#include <optional>
#include <string>
#include <variant>

#include <fmt/format.h>
#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"

#include "quantization.hpp"

namespace ttnn::operations::quantization {
namespace {

template <typename T>
void bind_quantize_operation(
    nb::module_& mod,
    const T& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16") {
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
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> input_tensor = ttnn.from_torch(torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scale = 0.001173
            >>> zero_point = -213
            >>> output = {1}(input_tensor, scale, zero_point)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor,
               const std::variant<ttnn::Tensor, float>& scale,
               const std::variant<Tensor, int32_t>& zero_point,
               const std::optional<int32_t> axis,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, scale, zero_point, axis, dtype, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("scale"),
            nb::arg("zero_point"),
            nb::kw_only(),
            nb::arg("axis") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("output_tensor") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

template <typename T>
void bind_requantize_operation(
    nb::module_& mod,
    const T& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16") {
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
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> input_tensor = ttnn.from_torch(torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> in_scale = 0.001173
            >>> in_zero_point = -213
            >>> out_scale = 0.002727
            >>> out_zero_point = -73
            >>> output = {1}(input_tensor, in_scale, in_zero_point, out_scale, out_zero_point)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor,
               const std::variant<ttnn::Tensor, float>& in_scale,
               const std::variant<Tensor, int32_t>& in_zero_point,
               const std::variant<ttnn::Tensor, float>& out_scale,
               const std::variant<Tensor, int32_t>& out_zero_point,
               const std::optional<int32_t> axis,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    in_scale,
                    in_zero_point,
                    out_scale,
                    out_zero_point,
                    axis,
                    dtype,
                    memory_config,
                    output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("in_scale"),
            nb::arg("in_zero_point"),
            nb::arg("out_scale"),
            nb::arg("out_zero_point"),
            nb::kw_only(),
            nb::arg("axis") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("output_tensor") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

template <typename T>
void bind_dequantize_operation(
    nb::module_& mod,
    const T& operation,
    const std::string& description,
    const std::string& supported_dtype = "BFLOAT16") {
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
            Supported dtypes, layouts, and ranks:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
                 - Ranks
               * - {3}
                 - TILE
                 - 2, 3, 4

            bfloat8_b/bfloat4_b supports only on TILE_LAYOUT

        Example:
            >>> input_tensor = ttnn.from_torch(torch.tensor([[-127 -42], [43 127]], dtype=torch.int32), layout=ttnn.TILE_LAYOUT, device=device)
            >>> scale = 0.001173
            >>> zero_point = -213
            >>> output = {1}(input_tensor, scale, zero_point)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name(),
        description,
        supported_dtype);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const T& self,
               const ttnn::Tensor& input_tensor,
               const std::variant<ttnn::Tensor, float>& scale,
               const std::variant<Tensor, int32_t>& zero_point,
               const std::optional<int32_t> axis,
               const std::optional<const DataType>& dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, scale, zero_point, axis, dtype, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("scale"),
            nb::arg("zero_point"),
            nb::kw_only(),
            nb::arg("axis") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("output_tensor") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace

void py_module(nb::module_& mod) {
    bind_quantize_operation(mod, ttnn::quantize, "Quantize Operation");
    bind_requantize_operation(mod, ttnn::requantize, "Re-quantize Operation");
    bind_dequantize_operation(mod, ttnn::dequantize, "De-quantize Operation");
}
}  // namespace ttnn::operations::quantization
