// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/types.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_sharded_to_interleaved(
    pybind11::module& module, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType>& output_dtype) -> ttnn::Tensor {
                return self(
                    input_tensor, memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG), output_dtype);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_dtype") = std::nullopt,
        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_sharded_to_interleaved(pybind11::module& module) {
    detail::bind_sharded_to_interleaved(
        module,
        ttnn::sharded_to_interleaved,
        R"doc(
        Converts a tensor from sharded to interleaved memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`memory_config` (ttnn.MemoryConfig): Memory configuration for the operation, must be Interleaved.

        Keyword Args:
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)

        )doc");
}

}  // namespace ttnn::operations::data_movement
