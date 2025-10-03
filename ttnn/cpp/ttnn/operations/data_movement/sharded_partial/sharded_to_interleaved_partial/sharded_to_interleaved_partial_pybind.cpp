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
               const std::optional<ttnn::DataType>& output_dtype) -> ttnn::Tensor {
                return self(input_tensor, cache_tensor, num_slices, slice_index, memory_config, output_dtype);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("cache_tensor").noconvert(),
            py::arg("num_slices"),
            py::arg("slice_index"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_dtype") = std::nullopt,
        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_sharded_to_interleaved_partial(pybind11::module& module) {
    detail::bind_sharded_to_interleaved_partial(
        module,
        ttnn::sharded_to_interleaved_partial,
        R"doc(
        Converts a partial tensor from sharded to interleaved memory layout.

        This operation writes a sharded tensor slice into a cache tensor at a specified slice index.

        Args:
            input_tensor (ttnn.Tensor): Input tensor in sharded memory layout.
            cache_tensor (ttnn.Tensor): Cache tensor to write the slice into.
            num_slices (int): Number of slices the cache tensor is divided into.
            slice_index (int): Index of the slice to write (0-indexed).

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the output. Defaults to input memory config.
            output_dtype (Optional[ttnn.DataType]): Output data type. Defaults to same as input.

        Returns:
            ttnn.Tensor: Output tensor in interleaved memory layout with the slice updated.

        Example:

            >>> interleaved_tensor = ttnn.sharded_to_interleaved_partial(tensor, cache_tensor, num_slices=4, slice_index=2)
        )doc");
}

}  // namespace ttnn::operations::data_movement
