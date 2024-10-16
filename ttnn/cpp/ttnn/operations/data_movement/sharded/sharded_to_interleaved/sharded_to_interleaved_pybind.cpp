// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "sharded_to_interleaved.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_sharded_to_interleaved(pybind11::module& module,
                                 const data_movement_sharded_operation_t& operation,
                                 const char* doc) {
    bind_registered_operation(module,
                              operation,
                              doc,
                              ttnn::pybind_overload_t{
                                  [](const data_movement_sharded_operation_t& self,
                                     const ttnn::Tensor& input_tensor,
                                     const std::optional<MemoryConfig>& memory_config,
                                     const std::optional<DataType>& output_dtype,
                                     uint8_t queue_id) -> ttnn::Tensor {
                                      return self(queue_id,
                                                  input_tensor,
                                                  memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
                                                  output_dtype);
                                  },
                                  py::arg("input_tensor").noconvert(),
                                  py::arg("memory_config") = std::nullopt,
                                  py::arg("output_dtype") = std::nullopt,
                                  py::kw_only(),
                                  py::arg("queue_id") = 0,
                              });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_sharded_to_interleaved(pybind11::module& module) {
    detail::bind_sharded_to_interleaved(
        module,
        ttnn::sharded_to_interleaved,
        R"doc(sharded_to_interleaved(input_tensor: ttnn.Tensor,  memory_config: ttnn.MemoryConfig, *,  queue_id: int) -> ttnn.Tensor

        Converts a tensor from sharded to interleaved memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`memory_config` (ttnn.MemoryConfig): Memory configuration for the operation, must be Interleaved.

        Keyword Args:
            * :attr:`queue_id`: command queue id
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)

        )doc");
}

}  // namespace ttnn::operations::data_movement
