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
               const std::optional<bool>& keep_l1_aligned) -> ttnn::Tensor {
                return self(
                    input_tensor, grid, shard_shape, shard_scheme, shard_orientation, output_dtype, keep_l1_aligned);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("grid"),
            py::arg("shard_shape"),
            py::arg("shard_scheme"),
            py::arg("shard_orientation"),
            py::arg("output_dtype") = std::nullopt,
            py::kw_only(),
            py::arg("keep_l1_aligned") = false,

        },
        ttnn::pybind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MemoryConfig& sharded_memory_config,
               const std::optional<ttnn::DataType>& output_dtype,
               const std::optional<bool>& keep_l1_aligned) -> ttnn::Tensor {
                return self(input_tensor, sharded_memory_config, output_dtype, keep_l1_aligned);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("sharded_memory_config"),
            py::arg("output_dtype") = std::nullopt,
            py::kw_only(),
            py::arg("keep_l1_aligned") = false,

        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void py_bind_interleaved_to_sharded(pybind11::module& module) {
    detail::bind_interleaved_to_sharded(
        module,
        ttnn::interleaved_to_sharded,
        R"doc(
        Converts a tensor from interleaved to sharded memory layout.

        Args:
            input_tensor (ttnn.Tensor): Input tensor in interleaved memory layout.
            grid (ttnn.CoreGrid or ttnn.CoreRangeSet): Grid of cores for sharding.
            shard_shape (List[int]): Shape of each shard as [height, width].
            shard_scheme (ttnn.TensorMemoryLayout): Sharding scheme (HEIGHT_SHARDED, WIDTH_SHARDED, or BLOCK_SHARDED).
            shard_orientation (ttnn.ShardOrientation): Shard orientation (ROW_MAJOR or COL_MAJOR).

        Keyword Args:
            output_dtype (Optional[ttnn.DataType]): Output data type. Defaults to same as input.
            keep_l1_aligned (bool): Whether to keep L1 memory aligned. Defaults to False.

        Returns:
            ttnn.Tensor: Output tensor in sharded memory layout.

        Example:

            >>> # Using grid, shape, scheme, and orientation
            >>> sharded_tensor = ttnn.interleaved_to_sharded(tensor, ttnn.CoreGrid(3, 3), [32, 32], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardOrientation.ROW_MAJOR)

            >>> # Using sharded memory config
            >>> shard_memory_config = ttnn.create_sharded_memory_config(
            ...     shape=(96, 96),
            ...     core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ...     strategy=ttnn.ShardStrategy.BLOCK
            ... )
            >>> sharded_tensor = ttnn.interleaved_to_sharded(tensor, shard_memory_config)
        )doc");
}

}  // namespace ttnn::operations::data_movement
