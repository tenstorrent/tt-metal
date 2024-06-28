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
        R"doc(pad(input_tensor: ttnn.Tensor, padding: Tuple[Tuple[int, int], ...], value: Union[int, float], *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
Pad tensor with constant value. Padded shape is accumulated if ttnn.pad is called on a tensor with padding.

Args:
    * :attr:`input_tensor`: input tensor
    * :attr:`padding`: padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor.
    * :attr:`value`: value to pad with
    * :attr:`queue_id` (Optional[uint8]): command queue id

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation)doc";

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
                const std::optional<ttnn::MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, padding, value, memory_config);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("padding"),
                py::arg("value"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::data_movement::detail
