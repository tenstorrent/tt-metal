// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_stats_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_stats/attn_res_stats.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_attn_res_stats(nb::module_& mod) {
    ttnn::bind_function<"attn_res_stats", "ttnn.experimental.">(
        mod,
        R"doc(
            Takes both rank-local `d`-reductions of an AttnRes read from a single
            pass over `v`, stacked for the statistics collective:

                out[c]     = sum_d v[c][n][d] * v[c][n][d]
                out[C + c] = sum_d v[c][n][d] * q[d]

            `v` is `[1, C, N, D]` and `q` is `[1, 1, 1, D]`, both in TILE layout;
            the output is `[1, 2C, N, 1]`, which is the layout `attn_res_scores`
            splits by page arithmetic on the far side of the collective.

            Unfused these are two independent streams of `v` — an RMSNorm
            statistics kernel and a matmul against `q` as a column — plus the
            slice, concat and typecast needed to bring their outputs together. A
            row of `v` is made resident once here and reduced twice, which bounds
            `D` by what fits in L1 alongside `q` and one transformed copy.
        )doc",
        &ttnn::experimental::reduction::attn_res_stats,
        nb::arg("v").noconvert(),
        nb::arg("q").noconvert(),
        nb::kw_only(),
        nb::arg("dtype").noconvert() = nb::none(),
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::reduction::detail
