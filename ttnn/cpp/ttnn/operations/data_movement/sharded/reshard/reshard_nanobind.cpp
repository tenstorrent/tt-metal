// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "reshard.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_sharded_operation_t>
void bind_reshard(nb::module_& mod, const data_movement_sharded_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const data_movement_sharded_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MemoryConfig& output_memory_config,
               const std::optional<Tensor>& output_tensor) -> ttnn::Tensor {
                return self(input_tensor, output_memory_config, output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("output_memory_config"),
            nb::arg("output_tensor").noconvert() = nb::none(),

        });
}

}  // namespace detail

// TODO: Add more descriptions to the arguments
void bind_reshard(nb::module_& mod) {
    detail::bind_reshard(
        mod,
        ttnn::reshard,
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

        )doc");
}

}  // namespace ttnn::operations::data_movement
