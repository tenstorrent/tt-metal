// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "interleaved_to_sharded_partial.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_interleaved_to_sharded_partial(
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
               int64_t& num_slices,
               int64_t& slice_index,
               tt::tt_metal::TensorMemoryLayout shard_scheme,
               tt::tt_metal::ShardOrientation shard_orientation,
               const std::optional<ttnn::DataType>& output_dtype,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    input_tensor,
                    grid,
                    shard_shape,
                    num_slices,
                    slice_index,
                    shard_scheme,
                    shard_orientation,
                    output_dtype);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("grid"),
            py::arg("shard_shape"),
            py::arg("num_slices"),
            py::arg("slice_index"),
            py::arg("shard_scheme"),
            py::arg("shard_orientation"),
            py::kw_only(),
            py::arg("output_dtype") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,

        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_interleaved_to_sharded_partial(pybind11::module& module) {
    detail::bind_interleaved_to_sharded_partial(
        module,
        ttnn::interleaved_to_sharded_partial,
        R"doc(interleaved_to_sharded_partial(input_tensor: ttnn.Tensor, grid: ttnn.CoreGrid,  num_slices: int, slice_index: int, shard_scheme: ttl.tensor.TensorMemoryLayout, shard_orientation: ttl.tensor.ShardOrientation,  *, output_dtype: Optional[ttnn.dtype] = None) -> ttnn.Tensor

        Converts a partial tensor from interleaved to sharded memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`grid` (ttnn.CoreGrid): Grid of sharded tensor
            * :attr:`num_slices` (int): Number of slices.
            * :attr:`slice_index` (int): Slice index.
            * :attr:`shard_scheme` (ttl.tensor.TensorMemoryLayout): Sharding scheme(height, width or block).
            * :attr:`shard_orienttion` (ttl.tensor.ShardOrientation): Shard orientation (ROW or COL major).

        Keyword Args:
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input.
            * :attr:`queue_id`: command queue id

        Example:

            >>> sharded_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.CoreGrid(3,3), 2, 2, ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.ShardOrientation.ROW_MAJOR)

        )doc");
}

}  // namespace ttnn::operations::data_movement
