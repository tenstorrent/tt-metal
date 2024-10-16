// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tilize.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_tilize(py::module &module) {
    auto doc =
        R"doc(
            Changes data layout of input tensor to TILE.

            Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                dtype (data type, optional): Data type of the output tensor. Defaults to `None`.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::tilize);
    ttnn::bind_registered_operation(module,
                                    ttnn::tilize,
                                    doc,
                                    ttnn::pybind_overload_t{
                                        [](const OperationType &self,
                                           const ttnn::Tensor &input_tensor,
                                           const std::optional<MemoryConfig> &memory_config,
                                           std::optional<DataType> output_dtype,
                                           bool use_multicore,
                                           uint8_t queue_id) {
                                            return self(
                                                queue_id, input_tensor, memory_config, output_dtype, use_multicore);
                                        },
                                        py::arg("input_tensor"),
                                        py::kw_only(),
                                        py::arg("memory_config") = std::nullopt,
                                        py::arg("dtype") = std::nullopt,
                                        py::arg("use_multicore") = true,
                                        py::arg("queue_id") = 0,
                                    });
}
}  // namespace ttnn::operations::data_movement::detail
