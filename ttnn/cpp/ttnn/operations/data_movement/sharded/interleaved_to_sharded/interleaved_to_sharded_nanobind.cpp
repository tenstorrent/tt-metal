// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "interleaved_to_sharded.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_interleaved_to_sharded(
    nb::module_& mod, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
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
            nb::arg("input_tensor").noconvert(),
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("shard_scheme"),
            nb::arg("shard_orientation"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false,

        },
        ttnn::nanobind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MemoryConfig& sharded_memory_config,
               const std::optional<ttnn::DataType>& output_dtype,
               const std::optional<bool>& keep_l1_aligned) -> ttnn::Tensor {
                return self(input_tensor, sharded_memory_config, output_dtype, keep_l1_aligned);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("sharded_memory_config"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false,

        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void bind_interleaved_to_sharded(nb::module_& mod) {
    detail::bind_interleaved_to_sharded(
        mod,
        ttnn::interleaved_to_sharded,
        R"doc(
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
