// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "interleaved_to_sharded.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
void bind_interleaved_to_sharded(nb::module_& mod) {
    const auto* doc =
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

            >>> sharded_tensor = ttnn.interleaved_to_sharded(tensor, ttnn.CoreGrid(3,3), [32,32], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.ShardOrientation.ROW_MAJOR)


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
            >>> sharded_tensor = ttnn.interleaved_to_sharded(tensor, shard_memory_config)

        )doc";

    // Bind the free function directly - no struct!
    ttnn::bind_function<"interleaved_to_sharded">(
        mod,
        doc,

        // Overload 1: Using grid, shard_shape, shard_scheme, shard_orientation (detailed)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const std::variant<CoreCoord, CoreRangeSet>&,
                std::array<uint32_t, 2>,
                TensorMemoryLayout,
                tt::tt_metal::ShardOrientation,
                const std::optional<ttnn::DataType>&,
                const std::optional<bool>&>(&ttnn::interleaved_to_sharded),
            nb::arg("input_tensor").noconvert(),
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("shard_scheme"),
            nb::arg("shard_orientation"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false),

        // Overload 2: Using MemoryConfig (simple)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const MemoryConfig&,
                const std::optional<ttnn::DataType>&,
                const std::optional<bool>&,
                const std::optional<ttnn::Tensor>&>(&ttnn::interleaved_to_sharded),
            nb::arg("input_tensor").noconvert(),
            nb::arg("sharded_memory_config"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("keep_l1_aligned") = false,
            nb::arg("preallocated_output") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
