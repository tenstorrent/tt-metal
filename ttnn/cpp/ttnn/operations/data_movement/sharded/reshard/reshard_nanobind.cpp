// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "reshard.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
void bind_reshard(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Converts a tensor from one sharded layout to another sharded layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`output_memory_config` (MemoryConfig): Memory config with shard spec of output tensor
            * :attr:`output_tensor` (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`, which allocates one.

        Example:
            >>> shape = (1, 1, 64, 64)
            >>> height_sharded = ttnn.create_sharded_memory_config(
            ...     shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
            ... )
            >>> block_sharded = ttnn.create_sharded_memory_config(
            ...     shape, core_grid=ttnn.CoreGrid(y=2, x=2), strategy=ttnn.ShardStrategy.BLOCK
            ... )
            >>> tensor = ttnn.from_torch(
            ...     torch.rand(shape, dtype=torch.bfloat16),
            ...     layout=ttnn.TILE_LAYOUT,
            ...     device=device,
            ...     memory_config=height_sharded,
            ... )
            >>> resharded = ttnn.reshard(tensor, block_sharded)

        )doc";

    ttnn::bind_function<"reshard">(
        mod,
        doc,
        &ttnn::reshard,
        nb::arg("input_tensor").noconvert(),
        nb::arg("output_memory_config"),
        nb::arg("output_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::data_movement
