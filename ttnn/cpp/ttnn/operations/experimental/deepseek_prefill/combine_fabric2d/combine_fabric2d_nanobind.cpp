// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "combine_fabric2d.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::detail {
void bind_experimental_combine_fabric2d_operation(nb::module_& mod) {
    ttnn::bind_function<"combine_fabric2d", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        MoE prefill combine over an explicitly-forwarded FABRIC_2D: expert-processed tokens go back to
        the chips they came from, chip-local DRAM -> eth -> destination chip's DRAM.

        Called exactly like `ttnn.experimental.deepseek_prefill.combine`, plus `expert_offsets`.

            dispatched_buffer     the tokens, one per page. A chip's page range for one expert holds
                                  that expert's tokens grouped by the chip they ORIGINATED on.
            dispatched_metadata   3 int32 per token: (linearized_coord, token_idx, topk_idx). The
                                  token's destination slot is page token_idx * num_experts_per_tok +
                                  topk_idx of the origin chip's output.
            expert_token_counts   tokens per expert over all origin chips; closes the last run.
            expert_region_offsets where each expert's region starts in dispatched_buffer.
            expert_offsets        where each ORIGIN chip's run starts inside each expert's region.
                                  Must be REPLICATED along the dispatch-group axis, since every chip
                                  needs every origin chip's boundaries for the experts it hosts.

        Returns the combined output, (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim) BFLOAT16
        ROW_MAJOR per device.

        BFLOAT16 ROW_MAJOR input only, and `init_zeros` must be false. There is no fp8 output path: fp8
        comes out of the packer during untilize and this op has no untilize stage.
        )doc",
        &combine_fabric2d,
        nb::arg("dispatched_buffer"),
        nb::arg("dispatched_metadata"),
        nb::arg("expert_token_counts"),
        nb::arg("expert_region_offsets"),
        nb::arg("expert_offsets"),
        nb::arg("dispatch_group_size"),
        nb::arg("experts_per_chip"),
        nb::arg("num_experts_per_tok"),
        nb::arg("seq_len_per_chip"),
        nb::arg("cluster_axis"),
        nb::arg("num_links"),
        nb::arg("topology"),
        nb::arg("memory_config"),
        nb::arg("init_zeros"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::detail
