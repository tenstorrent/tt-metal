// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "reshard.hpp"

namespace nb = nanobind;

namespace ttnn::operations::data_movement {

void bind_reshard(nb::module_& mod) {
    const char* doc = R"doc(
        Converts a tensor from one sharded layout to another sharded layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`output_memory_config` (MemoryConfig): Memory config with shard spec of output tensor

        Keyword args:
            * :attr:`output_tensor` (ttnn.Tensor, optional): preallocated output tensor

        Example:
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
            >>> sharded_tensor = ttnn.reshard(tensor, shard_memory_config)

        )doc";

    mod.def(
        "reshard",
        &ttnn::reshard,
        doc,
        nb::arg("input_tensor").noconvert(),
        nb::arg("output_memory_config"),
        nb::arg("output_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::data_movement
