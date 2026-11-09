// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_heads_decode_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/transformer/concat_heads_decode/concat_heads_decode.hpp"

namespace ttnn::operations::experimental::concat_heads_decode::detail {

void bind_concat_heads_decode(nb::module_& mod) {
    ttnn::bind_function<"concat_heads_decode", "ttnn.experimental.">(
        mod,
        R"doc(
            Shuffles [S=1, B=32, 32(num_heads), head_dim] tensor into tensor with shape [S=1, 1, B=32, num_heads * head_dim]. num_heads should be specified and be less than 32; the op will assume the input padded num_heads to 32 and will unpad it. The output is default width sharded by num heads.
        )doc",
        &ttnn::experimental::concat_heads_decode,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("num_heads").noconvert(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("sub_core_grids") = nb::none());
}

}  // namespace ttnn::operations::experimental::concat_heads_decode::detail
