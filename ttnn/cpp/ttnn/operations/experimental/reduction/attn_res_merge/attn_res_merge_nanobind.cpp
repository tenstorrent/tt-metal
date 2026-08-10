// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_merge_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"

#include "ttnn/operations/experimental/reduction/attn_res_merge/attn_res_merge.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_attn_res_merge(nb::module_& mod) {
    ttnn::bind_function<"attn_res_merge", "ttnn.experimental.">(
        mod,
        R"doc(
            Folds a live residual stream into a precomputed sealed-snapshot partial,
            in a single pass:

                m   = max(shift, live_scores)
                r   = exp(shift - m)
                lw  = exp(live_scores - m)
                out = (partial * r + prefix_sum * lw) / (mass * r + lw)

            `shift`, `mass` and `live_scores` carry one scalar per row and are
            broadcast along the last dim. They must be TILE layout with a logical
            last dim of 1, sharing one dtype; that is already the layout the hardware
            column broadcast reads, so no transpose or repeat is needed on the
            caller's side.

            `partial` and the scalar operands may each carry R read sites on dim 0,
            and `site` picks the plane; at R == 1 an operand is shared by every site
            and `site` does not apply to it. A batch can therefore be passed whole
            rather than sliced per site. The output is a single plane either way.

            `partial` and `prefix_sum` must be rank-4 bfloat16 with a candidate dim
            of 1, and `prefix_sum` is one plane: a single live stream sits behind
            every read site.

            Above zero `num_partials`, `live_scores` carries the statistics the live
            score comes from rather than the score — `[1, 2 * num_partials, N, 1]`,
            each rank's sum of squares then its dots, stacked rank-major the way a
            gathering collective leaves them — and this op sums the ranks and
            normalizes them itself:

                live_scores = dots * rsqrt(sum_squares * inv_hidden_size + eps)

            The chain runs in the same dest registers that already derive the row
            weights, so scoring separately costs a device program and a DRAM round
            trip and buys nothing. `inv_hidden_size` and `eps` are unread at zero.

            Equivalent to the eleven-op expression above without its full-size
            intermediates: the division folds into the row scalars, so the full-width
            work is two broadcast MACs and each operand is read exactly once.
        )doc",
        &ttnn::experimental::reduction::attn_res_merge,
        nb::arg("partial").noconvert(),
        nb::arg("prefix_sum").noconvert(),
        nb::arg("shift").noconvert(),
        nb::arg("mass").noconvert(),
        nb::arg("live_scores").noconvert(),
        nb::kw_only(),
        nb::arg("site") = 0,
        nb::arg("num_partials") = 0,
        nb::arg("inv_hidden_size") = 0.0f,
        nb::arg("eps") = 0.0f,
        nb::arg("memory_config").noconvert() = nb::none(),
        nb::arg("compute_kernel_config").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::reduction::detail
