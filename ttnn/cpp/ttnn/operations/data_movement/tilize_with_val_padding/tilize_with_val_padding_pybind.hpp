// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tilize_with_val_padding.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_tilize_with_val_padding(py::module &module) {
    auto doc =
        R"doc(
            tilize_with_val_padding(input_tensor: ttnn.Tensor, output_tensor_shape: ttnn.Shape, pad_value: Union[int, float], *, memory_config: Optional[MemoryConfig] = None, dtype: Optional[DataType] = None, use_multicore: bool = False, queue_id: int = 0) -> ttnn.Tensor

            Changes data layout of input tensor to TILE. Pads to specified shape with a user-provided value.

            Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`output_tensor_shape`: Shape of the output tensor.
                * :attr:`pad_value`: Value to pad the output tensor with.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor.
                * :attr:`dtype`: Data type of the output tensor.
                * :attr:`use_multicore`: Whether to use multicore.
                * :attr:`queue_id`: command queue id.
        )doc";

    using OperationType = decltype(ttnn::tilize_with_val_padding);
    ttnn::bind_registered_operation(
        module,
        ttnn::tilize_with_val_padding,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType &self,
               const ttnn::Tensor &input_tensor,
               const tt::tt_metal::Shape &output_tensor_shape,
               float value,
               const std::optional<MemoryConfig> &memory_config,
               std::optional<DataType> output_dtype,
               bool use_multicore,
               uint8_t queue_id) {
                return self(
                    queue_id, input_tensor, output_tensor_shape, value, memory_config, output_dtype, use_multicore);
            },
            py::arg("input_tensor"),
            py::arg("output_tensor_shape"),
            py::arg("pad_value"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("use_multicore") = false,
            py::arg("queue_id") = 0,
        });
}

void bind_tilize_with_zero_padding(py::module &module) {
    auto doc =
        R"doc(
            tilize_with_zero_padding(input_tensor: ttnn.Tensor, *, memory_config: Optional[MemoryConfig] = None, dtype: Optional[DataType] = None, use_multicore: bool = False, queue_id: int = 0) -> ttnn.Tensor

            Changes data layout of input tensor to TILE. Pads to the nearest multiple of TILE width/height with zero value.

            Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

            Args:
                * :attr:`input_tensor`: Input Tensor.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor.
                * :attr:`dtype`: Data type of the output tensor.
                * :attr:`use_multicore`: Whether to use multicore.
                * :attr:`queue_id`: command queue id.
        )doc";

    using OperationType = decltype(ttnn::tilize_with_zero_padding);
    ttnn::bind_registered_operation(
        module,
        ttnn::tilize_with_zero_padding,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType &self,
               const ttnn::Tensor &input_tensor,
               const std::optional<MemoryConfig> &memory_config,
               std::optional<DataType> output_dtype,
               bool use_multicore,
               uint8_t queue_id) { return self(queue_id, input_tensor, memory_config, output_dtype, use_multicore); },
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_dtype") = std::nullopt,
            py::arg("use_multicore") = false,
            py::arg("queue_id") = 0,
        });
}

}  // namespace ttnn::operations::data_movement::detail
