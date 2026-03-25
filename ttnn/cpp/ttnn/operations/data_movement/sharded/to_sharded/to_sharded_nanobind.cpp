// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_sharded_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/array.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "to_sharded.hpp"
#include "ttnn/types.hpp"
namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
// ALL update docstring to to_sharded
void bind_to_sharded(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Converts a tensor to a specified memory layout

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): input tensor
            * :attr:`memory_config` (MemoryConfig): Memory config for the output tensor.

        Keyword Args:
            * :attr:`output_dtype` (Optional[ttnn.DataType]): Output data type, defaults to same as input. If it is different from the input, then the tensor must be tilized.

        Example:
            >>> # Let's say we have an input_tensor with shape [3, 160, 160], and we want to convert it to an ND sharded tensor with shard shape [2, 64, 64] sharded over 4 cores:
            >>> shard_shape = [2, 64, 64]
            >>> grid = ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(3, 2), ttnn.CoreCoord(4, 2)),
                    }
                ),
            >>> shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
            >>> nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
            >>> nd_sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
            >>> nd_sharded_tensor = ttnn.to_sharded(input_tensor, nd_sharded_memory_config)
        )doc";

    ttnn::bind_function<"to_sharded">(
        mod,
        doc,
        // Using MemoryConfig (simple)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const MemoryConfig&,
                const std::optional<ttnn::DataType>&,
                const std::optional<ttnn::Tensor>&>(&ttnn::to_sharded),
            nb::arg("input_tensor").noconvert(),
            nb::arg("memory_config"),
            nb::arg("output_dtype") = nb::none(),
            nb::kw_only(),
            nb::arg("preallocated_output") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
