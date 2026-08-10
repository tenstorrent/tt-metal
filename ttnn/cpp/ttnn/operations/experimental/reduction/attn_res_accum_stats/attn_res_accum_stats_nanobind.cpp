// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_accum_stats_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_accum_stats/attn_res_accum_stats.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_attn_res_accum_stats(nb::module_& mod) {
    ttnn::bind_function<"attn_res_accum_stats", "ttnn.experimental.">(
        mod,
        R"doc(
            Accumulates a residual and takes both rank-local `d`-reductions of the
            result from a single pass, returning `(total, stats)`:

                total[c][n][d] = a[c][n][d] + b[c][n][d]
                stats[c]       = sum_d total[c][n][d] * total[c][n][d]
                stats[C + c]   = sum_d total[c][n][d] * q[d]

            `a` and `b` are bfloat16 `[1, C, N, D]` and `q` is `[1, 1, 1, D]`,
            all in TILE layout; `total` matches the addends and `stats` is
            `[1, 2C, N, 1]`, the layout the statistics collective expects.

            Unfused the sum is written to DRAM by one program and read straight
            back by the next, since a residual stream's next read consumes exactly
            what its accumulation just wrote. Here the sum is packed out of dest
            into both the reduce operand and the writer, so it never returns from
            DRAM. `D` is bounded by what fits in L1 alongside `q`, the resident
            sum and one transformed copy.
        )doc",
        &ttnn::experimental::reduction::attn_res_accum_stats,
        nb::arg("a").noconvert(),
        nb::arg("b").noconvert(),
        nb::arg("q").noconvert(),
        nb::kw_only(),
        nb::arg("stats_dtype").noconvert() = nb::none(),
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::reduction::detail
