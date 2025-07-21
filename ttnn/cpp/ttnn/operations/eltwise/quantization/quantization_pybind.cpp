// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization_pybind.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include "ttnn-pybind/decorators.hpp"

#include "quantization.hpp"

namespace ttnn::operations::quantization {
namespace {

template <typename T>
void bind_quantize_operation(
    py::module& module,
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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
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
            py::arg("input_tensor"),
            py::arg("scale"),
            py::arg("zero_point"),
            py::kw_only(),
            py::arg("axis") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename T>
void bind_requantize_operation(
    py::module& module,
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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
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
            py::arg("input_tensor"),
            py::arg("in_scale"),
            py::arg("in_zero_point"),
            py::arg("out_scale"),
            py::arg("out_zero_point"),
            py::kw_only(),
            py::arg("axis") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

template <typename T>
void bind_dequantize_operation(
    py::module& module,
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
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
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
            py::arg("input_tensor"),
            py::arg("scale"),
            py::arg("zero_point"),
            py::kw_only(),
            py::arg("axis") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace

void py_module(py::module& module) {
    bind_quantize_operation(module, ttnn::quantize, "Quantize Operation");
    bind_requantize_operation(module, ttnn::requantize, "Re-quantize Operation");
    bind_dequantize_operation(module, ttnn::dequantize, "De-quantize Operation");
}
}  // namespace ttnn::operations::quantization
