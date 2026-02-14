// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_partial_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "interleaved_to_sharded_partial.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement {

// TODO: Add more descriptions to the arguments
void bind_interleaved_to_sharded_partial(nb::module_& mod) {
    const auto* doc = R"doc(
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

        Example:

            >>> sharded_tensor = ttnn.sharded_to_interleaved(tensor, ttnn.CoreGrid(3,3), 2, 2, ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.ShardOrientation.ROW_MAJOR)

        )doc";

    ttnn::bind_function<"interleaved_to_sharded_partial">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const std::variant<CoreCoord, CoreRangeSet>&,
                const std::array<uint32_t, 2>&,
                int64_t&,
                int64_t&,
                tt::tt_metal::TensorMemoryLayout,
                tt::tt_metal::ShardOrientation,
                const std::optional<ttnn::DataType>&>(&ttnn::interleaved_to_sharded_partial),
            nb::arg("input_tensor").noconvert(),
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("num_slices"),
            nb::arg("slice_index"),
            nb::arg("shard_scheme"),
            nb::arg("shard_orientation"),
            nb::kw_only(),
            nb::arg("output_dtype") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
