// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_block_stats_nanobind.hpp"

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/attn_res_block_stats.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_stats::detail {

void bind_attn_res_block_stats(nb::module_& mod) {
    ttnn::bind_function<"attn_res_block_stats", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            This rank's AttnRes block statistics from one pass over the sealed set:

                out[0][c][n][0]     = sum_d v[c][n][d]^2
                out[0][c][n][1 + r] = sum_d v[c][n][d] * queries[0][0][d][1 + r]

            `sealed` is a list of `[1, S_i, N, d]` bfloat16 tensors whose planes, in order,
            are the C candidates. `queries` is `[1, 1, d, 32]` bfloat16 with column 0 zero.
            The output is `[1, C, N, 32]` float32, not yet summed across the
            tensor-parallel axis: the layout `attn_res_block_mix` takes after an
            all-gather. Blackhole only.

            Equivalent to `rms_norm_pre_all_gather(v)` plus `matmul(v, queries)` over the
            concatenated set, with one read of it instead of two and no concatenation.

            With `cluster_axis`, the op also gathers the statistics across that mesh axis:
            rank p's land in tile column p of `gathered`, `[1, C, N, 32 * P]` float32, which
            is returned. `gathered` must stay allocated across calls (peers write into it by
            address) and is the layout `attn_res_block_mix` reads.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::attn_res_block_stats::attn_res_block_stats,
        nb::arg("sealed"),
        nb::arg("queries").noconvert(),
        nb::kw_only(),
        nb::arg("gathered").noconvert() = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("topology") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_stats::detail
