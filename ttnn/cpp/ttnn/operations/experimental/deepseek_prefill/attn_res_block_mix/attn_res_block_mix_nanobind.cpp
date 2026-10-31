// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_block_mix_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/attn_res_block_mix.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix::detail {

void bind_attn_res_block_mix(nb::module_& mod) {
    ttnn::bind_function<"attn_res_block_mix", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Scores R AttnRes read sites against the sealed set and mixes it, returning
            `(partials, shift, mass)` in the online-softmax convention:

                s[r][c][n]       = stats[c][n][1 + r] * rsqrt(stats[c][n][0] * inv_hidden_size + eps)
                shift[r][0][n]   = max_c s[r][c][n]
                e[r][c][n]       = exp(s[r][c][n] - shift[r][0][n])
                mass[r][0][n]    = sum_c e[r][c][n]
                partials[r][0][n] = sum_c e[r][c][n] * v[c][n]

            `sealed` is a list of `[1, S_i, N, d]` bfloat16 tensors whose planes in order are
            the C candidates. `stats` is `[1, C, N, 32 * P]` float32: P tensor-parallel
            partials side by side, each with the sum of squares in column 0 and query r's
            dot in column 1 + r — `attn_res_block_stats` after an all-gather. `partials` is
            `[R, 1, N, d]` and `shift` and `mass` are `[R, 1, N, 1]` float32, the layouts
            `attn_res_gather_softmax` reads. The leading `num_closed_sites` sites have
            nothing live to fold in: their partials come out divided by their mass, which
            is the read's value. Blackhole only.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix::attn_res_block_mix,
        nb::arg("sealed"),
        nb::arg("stats").noconvert(),
        nb::kw_only(),
        nb::arg("num_sites"),
        nb::arg("inv_hidden_size"),
        nb::arg("eps"),
        nb::arg("num_closed_sites") = 0,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_mix::detail
