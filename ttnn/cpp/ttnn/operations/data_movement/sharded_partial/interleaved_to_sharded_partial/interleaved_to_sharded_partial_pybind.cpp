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
               const std::optional<ttnn::DataType>& output_dtype) -> ttnn::Tensor {
                return self(
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
        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_interleaved_to_sharded_partial(pybind11::module& module) {
    detail::bind_interleaved_to_sharded_partial(
        module,
        ttnn::interleaved_to_sharded_partial,
        R"doc(
        Converts a partial tensor from interleaved to sharded memory layout.

        This operation is useful for sharding only a portion of a tensor across cores.

        Args:
            input_tensor (ttnn.Tensor): Input tensor in interleaved memory layout.
            grid (ttnn.CoreGrid or ttnn.CoreRangeSet): Grid of cores for sharding.
            shard_shape (List[int]): Shape of each shard as [height, width].
            num_slices (int): Number of slices to divide the tensor into.
            slice_index (int): Index of the slice to shard (0-indexed).
            shard_scheme (ttnn.TensorMemoryLayout): Sharding scheme (HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED).
            shard_orientation (ttnn.ShardOrientation): Shard orientation (ROW_MAJOR or COL_MAJOR).

        Keyword Args:
            output_dtype (Optional[ttnn.DataType]): Output data type. Defaults to same as input.

        Returns:
            ttnn.Tensor: Output tensor in sharded memory layout containing the specified slice.

        Example:

            >>> sharded_tensor = ttnn.interleaved_to_sharded_partial(tensor, ttnn.CoreGrid(3, 3), [32, 32], num_slices=4, slice_index=2, shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, shard_orientation=ttnn.ShardOrientation.ROW_MAJOR)
        )doc");
}

}  // namespace ttnn::operations::data_movement
