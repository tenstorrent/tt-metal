// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "untilize_with_unpadding.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_untilize_with_unpadding(py::module &module) {
    auto doc =
        R"doc(
            Changes data layout of input tensor to ROW_MAJOR and unpads/removes elements from the tensor.

            Input tensor must be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            Args:
                input_tensor (ttnn.Tensor): the input tensor
                output_tensor_end (shape): End indices of input tensor in output tensor.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                use_pack_untilize (bool, optional): Whether to use pack untilize. Defaults to `True`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                List of ttnn.Tensor: the output tensor.
        )doc";

    using OperationType = decltype(ttnn::untilize_with_unpadding);
    ttnn::bind_registered_operation(
        module,
        ttnn::untilize_with_unpadding,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType &self,
               const ttnn::Tensor &input_tensor,
               const tt::tt_metal::LegacyShape &output_tensor_end,
               const std::optional<MemoryConfig> &memory_config,
               bool use_multicore,
               bool use_pack_untilize,
               uint8_t queue_id) {
                return self(queue_id, input_tensor, output_tensor_end, memory_config, use_multicore, use_pack_untilize);
            },
            py::arg("input_tensor"),
            py::arg("output_tensor_end"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("use_multicore") = false,
            py::arg("use_pack_untilize") = true,
            py::arg("queue_id") = 0,
        });
}

}  // namespace ttnn::operations::data_movement::detail
