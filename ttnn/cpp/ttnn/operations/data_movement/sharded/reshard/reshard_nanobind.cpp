// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "reshard.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
void bind_reshard(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Converts a tensor from one sharded layout to another sharded layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`output_memory_config` (MemoryConfig): Memory config with shard spec of output tensor

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

    ttnn::bind_function<"reshard">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::reshard,
            nb::arg("input_tensor").noconvert(),
            nb::arg("output_memory_config"),
            nb::arg("output_tensor").noconvert() = nb::none()));
}

}  // namespace ttnn::operations::data_movement
