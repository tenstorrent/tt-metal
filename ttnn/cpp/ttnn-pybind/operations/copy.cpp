// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy.hpp"

#include <optional>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::copy {

namespace {

void bind_global_typecast(py::module& module) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, dtype: ttnn.DataType, *, memory_config: Optional[ttnn.MemoryConfig] = None, output_tensor : Optional[ttnn.Tensor] = None, queue_id : Optional[int]) -> ttnn.Tensor

Applies {0} to :attr:`input_tensor`.

Args:
    * :attr:`input_tensor` (ttnn.Tensor): input tensors must be on device, in ROW MAJOR or TILE layout
    * :attr:`dtype` (Optional[ttnn.DataType]): data type must be one of the following types BFLOAT16,BFLOAT8_B,BFLOAT4_B,UINT32,INT32 and UINT16.
    *
Keyword Args:
    * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
    * :attr:`output_tensor` (Optional[ttnn.Tensor]): Preallocated tensor to store the output.

Returns:
    ttnn.Tensor: The tensor with the updated data type. Output tensor will be on device, in same layout, and have the given data type.

Example::

    >>> tensor = ttnn.typecast(torch.randn((10, 3, 32, 32), dtype=ttnn.bfloat16), ttnn.uint16)
)doc",
        ttnn::typecast.base_name());

    using TypecastType = decltype(ttnn::typecast);
    bind_registered_operation(
        module,
        ttnn::typecast,
        doc,
        ttnn::pybind_overload_t{
            [](const TypecastType& self,
               const ttnn::Tensor& input_tensor,
               const DataType dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, dtype, memory_config, output_tensor, sub_core_grids);
            },
            py::arg("input_tensor"),
            py::arg("dtype"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId},

        ttnn::pybind_overload_t{
            [](const TypecastType& self,
               const ttnn::Tensor& input_tensor,
               const DataType input_dtype,
               const DataType output_dtype,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id, input_tensor, input_dtype, output_dtype, memory_config, output_tensor, sub_core_grids);
            },
            py::arg("input_tensor"),
            py::arg("input_dtype"),
            py::arg("output_dtype"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId}

    );
}

}  // namespace

void py_module(py::module& module) { bind_global_typecast(module); }

}  // namespace ttnn::operations::copy
