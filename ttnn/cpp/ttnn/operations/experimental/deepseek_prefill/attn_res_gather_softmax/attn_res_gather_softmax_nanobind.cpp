// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_gather_softmax_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/attn_res_gather_softmax.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_gather_softmax::detail {

void bind_attn_res_gather_softmax(nb::module_& mod) {
    ttnn::bind_function<"attn_res_gather_softmax", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            One read site's whole path from a tensor-parallel-sharded residual stream
            to the mixed hidden state, in a single dispatch: the live stream's
            statistics, their exchange across `cluster_axis`, and the online-softmax
            fold against a precomputed sealed-snapshot partial.

            Each rank holds `d / ring_size` of every token, so the live score needs a
            dot and a sum of squares summed across the ring before it can be scaled:

                live_scores = sum_p dots_p
                              * rsqrt(sum_p sum_squares_p * inv_hidden_size + eps)
                m   = max(shift, live_scores)
                h   = (partial * exp(shift - m) + running_sum * exp(live_scores - m))
                      / (mass * exp(shift - m) + exp(live_scores - m))

            The wire carries the per-rank statistics rather than their reduction:
            two scalars per token cross either way, and summing them on arrival is a
            pair of dest-register adds inside a pass that already holds those tiles,
            while reducing on the wire would cost a second device program.

            `partial` and `running_sum` are rank-4 bfloat16 with a candidate dim of 1,
            and `running_sum` is a single plane: one live stream sits behind every
            read site. `shift` and `mass` carry one scalar per token row in TILE
            layout with a logical last dim of 1.

            `stats` is caller-allocated scratch shaped `[1, 2 * ring_size, N, 1]` in
            the same dtype as `shift` and `mass`. It is not read on entry and holds
            nothing meaningful on exit; it is an operand so that a caller walking
            many read sites allocates it once. It must be allocated across the whole
            mesh — the exchange addresses a peer's slot by page of that peer's own
            copy.

            `partial`, `shift` and `mass` may each carry R read sites on dim 0, with
            `site` picking the plane; at R == 1 an operand is shared by every site
            and `site` does not apply to it. `site` shapes no kernel, so a walk over
            R sites reuses one cached program.

            `pending` is a residual write not yet in the stream. Given, the op scores
            and folds `running_sum + pending` and returns that sum alongside `h` for
            the caller to carry forward, which is a whole dispatch cheaper than
            `ttnn.add`-ing it first. The return is a list of one tensor without it and
            two with it.

            Blackhole only, and requires a ring size above 1 on `cluster_axis`.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::attn_res_gather_softmax::attn_res_gather_softmax,
        nb::arg("partial").noconvert(),
        nb::arg("running_sum").noconvert(),
        nb::arg("shift").noconvert(),
        nb::arg("mass").noconvert(),
        nb::arg("q").noconvert(),
        nb::arg("stats").noconvert(),
        nb::arg("semaphore"),
        nb::kw_only(),
        nb::arg("cluster_axis"),
        nb::arg("site") = 0,
        nb::arg("inv_hidden_size"),
        nb::arg("eps") = 0.0f,
        nb::arg("pending").noconvert() = nb::none(),
        nb::arg("num_links") = nb::none(),
        nb::arg("topology") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_gather_softmax::detail
