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

            dispatched_buffer     the tokens: BFLOAT16 ROW_MAJOR, BFLOAT16 TILE or BFLOAT8_B TILE. A
                                  chip's token range for one expert holds that expert's tokens grouped
                                  by the chip they ORIGINATED on.
            dispatched_metadata   3 int32 per token: (linearized_coord, token_idx, topk_idx). The
                                  token's destination slot is page token_idx * num_experts_per_tok +
                                  topk_idx of the origin chip's output.
            expert_token_counts   tokens per expert over all origin chips; closes the last run.
            expert_region_offsets where each expert's region starts in dispatched_buffer.
            expert_offsets        where each ORIGIN chip's run starts inside each expert's region.
                                  Must be REPLICATED along the dispatch-group axis, since every chip
                                  needs every origin chip's boundaries for the experts it hosts.

        dispatched_buffer and dispatched_metadata are 1 in every dim before the last two, as dispatch
        leaves them, so their second-to-last dim is dispatch's max_dispatch_buffer_token_size. A token
        dispatch dropped for want of room, one whose page falls past it, is skipped. Its output slot,
        like that of a pick routed to another dispatch group, is not written and keeps whatever the
        output buffer held: unlike `combine`, the output is never zeroed, so mask those picks before
        reducing over top-k.

        Returns the combined output, (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim) BFLOAT16
        ROW_MAJOR per device.

        The output is BFLOAT16 whatever the input: a BFLOAT8_B buffer is dequantised by the untilize,
        exactly, since every bfp8 value is representable in BFLOAT16. There is no BFLOAT8_B ROW_MAJOR
        input (bfp8 only exists tiled).

        fp8_scale_block=128 (Blackhole, TILE dispatched_buffer) has the untilize quantize each token
        instead: e4m3 values, each 128 scaled by a power of two 2^ceil(log2(max(amax, 1e-4) / 448)) and
        rounded to nearest even, followed by those scales as fp32. The output is then
        (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim + emb_dim / 32) FP8_E4M3, one byte per
        element, which post_combine_reduce dequantizes with fp8_scale_block_w=128. A token row must be
        a multiple of 64 B, so emb_dim a multiple of 2048.
        )doc",
        &combine_fabric2d,
        nb::arg("dispatched_buffer"),
        nb::arg("dispatched_metadata"),
        nb::arg("expert_token_counts"),
        nb::arg("expert_region_offsets"),
        nb::arg("expert_offsets"),
        nb::arg("experts_per_chip"),
        nb::arg("num_experts_per_tok"),
        nb::arg("seq_len_per_chip"),
        nb::arg("cluster_axis"),
        nb::arg("num_links"),
        nb::arg("topology"),
        nb::arg("memory_config"),
        nb::arg("fp8_scale_block") = 0);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::detail
