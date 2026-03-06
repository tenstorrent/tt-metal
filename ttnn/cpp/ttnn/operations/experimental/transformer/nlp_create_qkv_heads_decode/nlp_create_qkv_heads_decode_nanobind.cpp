// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "nlp_create_qkv_heads_decode.hpp"

namespace ttnn::operations::experimental::transformer::detail {

void bind_nlp_create_qkv_heads_decode(nb::module_& mod) {
    ttnn::bind_function<"nlp_create_qkv_heads_decode", "ttnn.experimental.">(
        mod,
        R"doc(
            Shuffles [1, S=1, B, head_dim * (num_heads + 2*num_kv_heads)] fused qkv matrix into Q, K, and V heads with shape [S, B, num_heads, head_dim] for Q and [S, B, num_kv_heads, head_dim] for K and V, where num_heads and num_kv_heads will be padded to nearest 32. Input must be sharded, B=32 and S=1. If ttnn pads B from some number < 32 to 32, this op respects the unpadded B.
            overlap_qk_coregrid is a boolean flag that determines whether the output Q and K heads are on same core grid. If true, then Q, K, and V heads are on the same core grid. If false, the Q and K heads are on non-overlapping core-grid useful for processing Q and K in parallel.
        )doc",
        &ttnn::experimental::nlp_create_qkv_heads_decode,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("num_heads"),
        nb::arg("num_kv_heads") = nb::none(),
        nb::arg("output_tensors") = nb::none(),
        nb::arg("overlap_qk_coregrid") = true,
        nb::arg("batch_offset") = nb::none(),
        nb::arg("slice_size") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::transformer::detail
