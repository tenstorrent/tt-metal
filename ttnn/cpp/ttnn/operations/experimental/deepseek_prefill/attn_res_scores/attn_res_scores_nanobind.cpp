// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_scores_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_scores/attn_res_scores.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_scores::detail {

void bind_attn_res_scores(nb::module_& mod) {
    ttnn::bind_function<"attn_res_scores", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Scores R AttnRes read sites against a sealed snapshot set, from statistics
            already summed across the tensor-parallel axis, and returns the online-softmax
            pieces `(exponentials, shift, mass)`:

                s[r][c][n]       = stats[0][c][n][1 + r] * rsqrt(stats[0][c][n][0] * inv_hidden_size + eps)
                shift[r][0][n]   = max_c s[r][c][n]
                exponentials[r][c][n] = exp(s[r][c][n] - shift[r][0][n])
                mass[r][0][n]    = sum_c exponentials[r][c][n]

            `stats` is `[1, C, N, W]` float32 in TILE layout, with each candidate's sum of
            squares in column 0 and its dot with query r in column 1 + r, so `W` is at
            least `1 + num_sites` and at most one tile. The outputs are float32 with a
            logical last dim of 1: `exponentials` is `[R, C, N, 1]` and `shift` and
            `mass` are `[R, 1, N, 1]`, the layouts `attn_res_weighted_reduce_nc` and
            `attn_res_gather_softmax` read.

            Equivalent to the composite `mul`, `rsqrt`, `max`, `sub`, `exp`, `sum` and
            `permute` chain, in one dispatch. Blackhole only.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::attn_res_scores::attn_res_scores,
        nb::arg("stats").noconvert(),
        nb::kw_only(),
        nb::arg("num_sites"),
        nb::arg("inv_hidden_size"),
        nb::arg("eps"),
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_scores::detail
