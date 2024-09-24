// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "pad.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_pad(py::module& module) {
    auto doc =
        R"doc(

            Returns a padded tensor, with a specified value at the specified location. If the input tensor is on host, the pad will be performed on host, and if its on device it will be performed on device.

            Equivalent pytorch code:

            .. code-block:: python

                torch.pad(input_tensor, padding, value)
                torch.pad(input_tensor, output_tensor_shape, input_tensor_start, value)

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                padding (ttnn.Tensor): padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor. Mutually exclusive to output_tensor_shape and input_tensor_start.
                output_tensor_shape (shape): Final shape of padded tensor. This along with input_tensor_start is mutually exclusive from padding.
                input_tensor_start (shape): Shape describing where to start padding. This along with output_tensor_shape is mutually exclusive from padding.
                value (number): value to pad with.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
               List of ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::pad);
    ttnn::bind_registered_operation(
        module,
        ttnn::pad,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                std::vector<std::pair<uint32_t, uint32_t>> padding,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, padding, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("padding"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = true,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array1D & output_padded_shape,
                const tt::tt_metal::Array1D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array2D & output_padded_shape,
                const tt::tt_metal::Array2D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array3D & output_padded_shape,
                const tt::tt_metal::Array3D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array4D & output_padded_shape,
                const tt::tt_metal::Array4D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array5D & output_padded_shape,
                const tt::tt_metal::Array5D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array6D & output_padded_shape,
                const tt::tt_metal::Array6D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array7D & output_padded_shape,
                const tt::tt_metal::Array7D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                },
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const tt::tt_metal::Array8D & output_padded_shape,
                const tt::tt_metal::Array8D & input_tensor_start,
                const float value,
                const bool use_multicore,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config);
                },
                py::arg("input_tensor"),
                py::arg("output_padded_shape"),
                py::arg("input_tensor_start"),
                py::arg("value"),
                py::kw_only(),
                py::arg("use_multicore") = false,
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0,
                }
        );
}
}  // namespace ttnn::operations::data_movement::detail
