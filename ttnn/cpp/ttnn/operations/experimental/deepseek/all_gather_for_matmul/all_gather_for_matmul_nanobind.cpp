// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_for_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/all_gather_for_matmul/all_gather_for_matmul.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::all_gather_for_matmul::detail {

void bind_all_gather_for_matmul(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Gather a WIDTH_SHARDED input, or multicast a single-core HEIGHT_SHARDED input, for matmul.

        WIDTH_SHARDED: each input core untilizes its width shard and sends it to a hub. The hub
        assembles the complete input tensor and broadcasts one full row-major copy to every core
        in ``output_core_range_set``.

        HEIGHT_SHARDED: the full tensor already lives on one core, so that core untilizes if needed
        and multicasts the replica (no gather). TILE and ROW_MAJOR inputs are accepted.

        The output is HEIGHT_SHARDED in L1 and its shard shape is the logical shape of the complete
        input tensor (tile padding is dropped), matching matmul_decode full-width hub mode for A.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): L1 TILE WIDTH_SHARDED input, or a single-core
              HEIGHT_SHARDED TILE / ROW_MAJOR input
            * :attr:`output_core_range_set` (CoreRangeSet): receiver cores for the output

        Returns:
            ttnn.Tensor: row-major height-sharded tensor with one complete input replica per core
        )doc";

    ttnn::bind_function<"all_gather_for_matmul", "ttnn.experimental.deepseek.">(
        mod,
        doc,
        &ttnn::experimental::deepseek::all_gather_for_matmul,
        nb::arg("input_tensor").noconvert(),
        nb::arg("output_core_range_set"),
        nb::arg("output_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek::all_gather_for_matmul::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_all_gather_for_matmul(::nanobind::module_& mod) {
    all_gather_for_matmul::detail::bind_all_gather_for_matmul(mod);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
