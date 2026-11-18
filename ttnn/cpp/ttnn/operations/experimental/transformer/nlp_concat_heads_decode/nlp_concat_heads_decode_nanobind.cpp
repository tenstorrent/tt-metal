// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_concat_heads_decode_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads_decode::detail {

void bind_nlp_concat_heads_decode(nb::module_& mod) {
    ttnn::bind_function<"nlp_concat_heads_decode", "ttnn.experimental.">(
        mod,
        R"doc(
            Concatenates heads into [1, 1, 32, num_heads * padded_head_dim] by default.
            The input has shape [1, users, padded_heads, head_dim]; padded_head_dim is its padded width.
            The input must be L1 HEIGHT_SHARDED with one [padded_heads, padded_head_dim] shard per user,
            at most 32 users, and padded_heads a multiple of 32 containing num_heads.
            Input and output tiles must be non-transposed 32x32 with 16x16 faces. BF16 and FP32
            are supported. Shard orientation may be ROW_MAJOR or COL_MAJOR.

            output_tensor, when supplied, must be on the input device, have matching dtype,
            TILE layout, and L1 WIDTH_SHARDED memory. Its padded shape must match the result,
            it must contain every user row and have logical width num_heads * padded_head_dim.
            At least num_heads cores are required, each with a [32, padded_head_dim] shard.
            Spare cores are unused; for one head, the supplied output may instead keep the input logical width.

            sub_core_grids provides the available output cores and is required for non-origin or
            disjoint input grids. Without output_tensor, the output is L1 WIDTH_SHARDED with
            one core per head. memory_config is currently ignored.
        )doc",
        &ttnn::experimental::nlp_concat_heads_decode,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("num_heads").noconvert(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads_decode::detail
