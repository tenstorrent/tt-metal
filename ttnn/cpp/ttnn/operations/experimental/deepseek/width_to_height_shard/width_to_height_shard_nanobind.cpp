// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "width_to_height_shard_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/width_to_height_shard/width_to_height_shard.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::width_to_height_shard::detail {

void bind_width_to_height_shard(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Broadcast a width-sharded tensor onto a height-sharded layout.

        Each input core untilizes its width shard and sends it to a hub. The hub assembles the
        complete input tensor and broadcasts one full row-major copy to every core in
        ``output_core_range_set``. The output is HEIGHT_SHARDED in L1 and its shard shape is the
        logical shape of the complete input tensor (tile padding is dropped).

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): TILE-layout WIDTH_SHARDED L1 input
            * :attr:`output_core_range_set` (CoreRangeSet): receiver cores for the output

        Returns:
            ttnn.Tensor: row-major height-sharded tensor with one complete input replica per core
        )doc";

    ttnn::bind_function<"width_to_height_shard", "ttnn.experimental.deepseek.">(
        mod,
        doc,
        &ttnn::experimental::deepseek::width_to_height_shard,
        nb::arg("input_tensor").noconvert(),
        nb::arg("output_core_range_set"),
        nb::arg("output_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek::width_to_height_shard::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_width_to_height_shard(::nanobind::module_& mod) {
    width_to_height_shard::detail::bind_width_to_height_shard(mod);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
