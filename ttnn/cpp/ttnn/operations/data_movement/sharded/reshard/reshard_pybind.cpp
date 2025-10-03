// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "reshard.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_reshard(pybind11::module& module, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MemoryConfig& output_memory_config,
               const std::optional<Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, output_memory_config, output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("output_memory_config"),
            py::arg("output_tensor").noconvert() = std::nullopt,

        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_reshard(pybind11::module& module) {
    detail::bind_reshard(
        module,
        ttnn::reshard,
        R"doc(
        Converts a tensor from one sharded layout to another sharded layout.

        This operation allows changing the sharding configuration of an already sharded tensor.

        Args:
            input_tensor (ttnn.Tensor): Input tensor in sharded memory layout.
            output_memory_config (ttnn.MemoryConfig): Memory configuration with shard spec for the output tensor.

        Keyword Args:
            output_tensor (Optional[ttnn.Tensor]): Preallocated output tensor. Defaults to None.

        Returns:
            ttnn.Tensor: Output tensor with the new sharding configuration.

        Example:

            >>> shard_memory_config = ttnn.create_sharded_memory_config(
            ...     shape=input_shape,
            ...     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ...     strategy=ttnn.ShardStrategy.BLOCK
            ... )
            >>> resharded_tensor = ttnn.reshard(sharded_tensor, shard_memory_config)
        )doc");
}

}  // namespace ttnn::operations::data_movement
