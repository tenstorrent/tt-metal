// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_scores_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_scores/attn_res_scores.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_attn_res_scores(nb::module_& mod) {
    ttnn::bind_function<"attn_res_scores", "ttnn.experimental.">(
        mod,
        R"doc(
            Turns the globally summed AttnRes statistics into the scores themselves,
            in a single pass:

                scores[c] = dots[c] * rsqrt(sum_squares[c] * inv_hidden_size + eps)

            `stats` is `[1, 2C, N, W]` in TILE layout: the two statistics stacked on
            dim 1 so that one collective covers both. Candidates `[0, C)` hold the
            sums of squares and `[C, 2C)` the dots, and the output is `[1, C, N, W]`.

            Splitting the pair is page arithmetic rather than two `slice` calls, and
            no intermediate is materialized — the scale, the epsilon, the reciprocal
            square root and the multiply all run in dest registers, so the result
            rounds to `dtype` once.
        )doc",
        &ttnn::experimental::reduction::attn_res_scores,
        nb::arg("stats").noconvert(),
        nb::arg("inv_hidden_size"),
        nb::arg("eps"),
        nb::kw_only(),
        nb::arg("dtype").noconvert() = nb::none(),
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::reduction::detail
