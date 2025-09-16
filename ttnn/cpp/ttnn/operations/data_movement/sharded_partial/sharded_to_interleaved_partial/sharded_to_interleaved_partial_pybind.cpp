// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "sharded_to_interleaved_partial.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_sharded_to_interleaved_partial(
    pybind11::module& module, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& cache_tensor,
               int64_t& num_slices,
               int64_t& slice_index,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::DataType>& output_dtype,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input_tensor, cache_tensor, num_slices, slice_index, memory_config, output_dtype);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("cache_tensor").noconvert(),
            py::arg("num_slices"),
            py::arg("slice_index"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_dtype") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,

        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_sharded_to_interleaved_partial(pybind11::module& module) {
    detail::bind_sharded_to_interleaved_partial(
        module,
        ttnn::sharded_to_interleaved_partial,
        R"doc(sharded_to_interleaved_partial(input_tensor: ttnn.Tensor, cache_tensor: ttnn.Tensor,  num_slices: int, slice_index: int, *, output_dtype: Optional[ttnn.dtype] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Converts a partial tensor from sharded_to_interleaved memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`cache_tensor` (ttnn.Tensor): cache tensor
            * :attr:`num_slices` (int): Number of slices.
            * :attr:`slice_index` (int): Slice index.

        Keyword Args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.
            * :attr:`queue_id`: command queue id

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved(tensor, cache_tensor, 4, 2)

        )doc");
}

}  // namespace ttnn::operations::data_movement
