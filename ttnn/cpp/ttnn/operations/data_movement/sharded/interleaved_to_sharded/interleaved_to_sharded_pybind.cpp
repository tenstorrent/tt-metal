// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "interleaved_to_sharded.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_interleaved_to_sharded(
    pybind11::module& module, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::variant<CoreCoord, CoreRangeSet>& grid,
               const std::array<uint32_t, 2>& shard_shape,
               tt::tt_metal::TensorMemoryLayout shard_scheme,
               tt::tt_metal::ShardOrientation shard_orientation,
               const std::optional<ttnn::DataType>& output_dtype,
               QueueId queue_id,
               const std::optional<bool>& keep_l1_aligned) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    grid,
                    shard_shape,
                    shard_scheme,
                    shard_orientation,
                    output_dtype,
                    keep_l1_aligned);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("grid"),
            py::arg("shard_shape"),
            py::arg("shard_scheme"),
            py::arg("shard_orientation"),
            py::arg("output_dtype") = std::nullopt,
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId,
            py::arg("keep_l1_aligned") = false,

        },
        ttnn::pybind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MemoryConfig& sharded_memory_config,
               const std::optional<ttnn::DataType>& output_dtype,
               QueueId queue_id,
               const std::optional<bool>& keep_l1_aligned) -> ttnn::Tensor {
                return self(queue_id, input_tensor, sharded_memory_config, output_dtype, keep_l1_aligned);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("sharded_memory_config"),
            py::arg("output_dtype") = std::nullopt,
            py::kw_only(),
            py::arg("queue_id") = DefaultQueueId,
            py::arg("keep_l1_aligned") = false,

        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_interleaved_to_sharded(pybind11::module& module) {
    detail::bind_interleaved_to_sharded(
        module,
        ttnn::interleaved_to_sharded,
        R"doc(interleaved_to_sharded(input_tensor: ttnn.Tensor, grid: ttnn.CoreGrid,  int, shard_shape: List[int[2]], shard_scheme: ttl.tensor.TensorMemoryLayout, shard_orientation: ttl.tensor.ShardOrientation, sharded_memory_config: MemoryConfig *, output_dtype: Optional[ttnn.dtype] = None) -> ttnn.Tensor

        Converts a tensor from interleaved to sharded memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`grid` (ttnn.CoreGrid): Grid of sharded tensor
            * :attr:`shard_shape` (List(int[2])): Sharding shape.
            * :attr:`shard_scheme` (ttl.tensor.TensorMemoryLayout): Sharding scheme(height, width or block).
            * :attr:`shard_orientation` (ttl.tensor.ShardOrientation): Shard orientation (ROW or COL major).
            * :attr:`sharded_memory_config` (MemoryConfig): Instead of shard_shape, shard_scheme and orientation you can provide a single MemoryConfig representing the sharded tensor.

        Keyword Args:
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.
            * :attr:`queue_id`: command queue id

        Example 1 (using grid, shape, scheme, orienttion):

            >>> sharded_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.CoreGrid(3,3), [32,32], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.ShardOrientation.ROW_MAJOR)


        Example 2 (using sharded memory config):
            >>> sharded_memory_config_dict = dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)
                        ),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            >>> shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)
            >>> sharded_tensor = ttnn.sharded_to_interleaved(tensor, shard_memory_config)

        )doc");
}

}  // namespace ttnn::operations::data_movement
