// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dispatch_fabric2d.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d::detail {
void bind_experimental_dispatch_fabric2d_operation(nb::module_& mod) {
    ttnn::bind_function<"dispatch_fabric2d", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        MoE prefill dispatch over an explicitly-forwarded FABRIC_2D: each token goes to the chips
        hosting the experts it was routed to.

        Called like `ttnn.experimental.deepseek_prefill.dispatch`, with two differences: `expert_offsets`
        is the all-rows table rather than this device's row, and there is no weights, padding_config or
        scales input.

            input_tensor          the tokens, one per page, BFLOAT16 ROW_MAJOR.
            indices_tensor        top-k expert ids per token, UINT16 ROW_MAJOR.
            expert_offsets        where each SOURCE chip's run starts inside each expert's region, for
                                  every source chip: offset_cumsum's all_global_dispatch_offsets. Must
                                  be REPLICATED along the dispatch axis, because a relaying chip sizes a
                                  run it neither wrote nor receives.
            expert_dispatch_table global expert id -> chip in the dispatch group, -1 when the expert is
                                  not in this group.
            expert_token_counts   tokens per expert summed over every source chip.
            expert_region_offsets where each expert's region starts in the destination buffer. Together
                                  with expert_token_counts this closes the last source chip's run, which
                                  expert_offsets alone cannot: its rows are absolute buffer positions, so
                                  the close is counts + region_offsets - row, not counts alone.

        Returns {dispatched_buffer, metadata}, both per device and ROW_MAJOR:
        dispatched_buffer is (1, 1, max_dispatch_buffer_token_size, emb_dim) BFLOAT16 and metadata is
        (1, 1, max_dispatch_buffer_token_size, 3) INT32 carrying (src chip, token index, top-k slot) at
        the same page index as the token.

        A token whose expert region is full is dropped while its counter still advances, so pages match
        what `dispatch` would have assigned.

        BFLOAT16 ROW_MAJOR input only, and metadata_len must be 3: the fp8-scaled layout appends
        per-block scales that do not fit the routing tail this op carries.
        )doc",
        &dispatch_fabric2d,
        nb::arg("input_tensor"),
        nb::arg("indices_tensor"),
        nb::arg("expert_offsets"),
        nb::arg("expert_dispatch_table"),
        nb::arg("expert_token_counts"),
        nb::arg("expert_region_offsets"),
        nb::arg("experts_per_chip"),
        nb::arg("num_routed_experts"),
        nb::arg("num_experts_per_tok"),
        nb::arg("metadata_len"),
        nb::arg("max_dispatch_buffer_token_size"),
        nb::arg("seq_len_per_chip"),
        nb::arg("cluster_axis"),
        nb::arg("num_links"),
        nb::arg("topology"),
        nb::arg("memory_config"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d::detail
