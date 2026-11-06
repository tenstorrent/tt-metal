// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_and_split_heads_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/untilize_and_split_heads/untilize_and_split_heads.hpp"

namespace ttnn::operations::experimental::untilize_and_split_heads::detail {

void bind_untilize_and_split_heads(nb::module_& mod) {
    const auto* doc =
        R"doc(
        (B, S, H*D) TILE -> (B, H, ceil(S/32), 32*D) ROW_MAJOR: splits the heads
        of a tiled tensor, as a Linear emits it, and untilizes them in one pass,
        one tile row of a head per output page::

            output[b, h, t, r*D + d] = input[b, 32*t + r, h*D + d]

        The output holds the bytes of a row-major (B, H, ceil(S/32)*32, D) tensor,
        but each page is one untilized tile row of a head: 32 consecutive rows, so
        a consumer reads neighbouring rows of a head within a tile row in one NoC
        read. ttnn.experimental.fused_msda and fused_msda_from_offsets read it with
        value_layout="head_major".

        Args:
            * :attr:`input`: (B, S, H*D) TILE bfloat16, INTERLEAVED; D a positive
              multiple of 32; non-transposed 32x32 tiles padded only up to a whole tile
              of S, as ttnn.linear and ttnn.to_layout produce
            * :attr:`num_heads`: H
            * :attr:`memory_config`: output memory config, INTERLEAVED; None keeps
              input's

        Returns:
            (B, H, ceil(S/32), 32*D) ROW_MAJOR bfloat16. Rows past S in each head's
            last page are unspecified: they hold whatever the input's tile padding
            holds, which need not be zero.
        )doc";

    ttnn::bind_function<"untilize_and_split_heads", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::untilize_and_split_heads,
        nb::arg("input"),
        nb::arg("num_heads"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::untilize_and_split_heads::detail
