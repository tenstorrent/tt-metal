// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routing_metadata_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "routing_metadata.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

void bind_routing_metadata(nb::module_& mod) {
    ttnn::bind_function<"routing_metadata", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        Builds the metadata of a routed top-k MoE pass from the router's selection, on device.

        A routed pass copies each token's activation row once for each of its top_k experts into
        a gathered matrix, sorted by expert, so a grouped matmul runs every expert only on its own
        rows. Each expert's region starts on a 32-row block, in expert order, with its tokens in
        increasing order. The outputs say where every row goes and how the results combine back.
        Nothing is read back to the host, so the call is trace-safe.

        With W = num_tokens, gathered_rows = 32 * (ceil(top_k * W / 32) + num_experts - 1), which
        holds any routing, and padded_tokens = 32 * ceil(W / 32).

        Args:
            indices (ttnn.Tensor): (1, 1, tokens, top_k) UINT32, TILE, interleaved: the expert ids
                from ttnn.topk. An id at or above num_experts counts as num_experts - 1.
            values (ttnn.Tensor): (1, 1, tokens, top_k) FLOAT32, TILE, interleaved: their weights.

        Keyword Args:
            num_experts (int): number of experts, 1 to 64.
            token_start (int): first token of the window, a multiple of 32. Defaults to 0.
            num_tokens (int, optional): tokens in the window, at most 65536 slots: 1 to
                65536 / top_k, 32768 at top-2. Defaults to the rest.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor], all in DRAM:

            gather_index: (1, gathered_rows) UINT32, ROW_MAJOR. The global token id each gathered
            row copies. Rows that hold no slot copy token_start.

            combine_index: (top_k, padded_tokens) UINT32, ROW_MAJOR. The gathered row of window
            token i's k-th choice; 0 for the padding tokens.

            combine_weights: (1, top_k, padded_tokens, 1) FLOAT32, TILE. values[token_start + i][k],
            bit-exact; 0 for the padding tokens.

            expert_counts: (1, num_experts) UINT32, ROW_MAJOR. Slots routed to each expert. Region
            offsets and each block's expert follow from these.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::routing_metadata,
        nb::arg("indices"),
        nb::arg("values"),
        nb::kw_only(),
        nb::arg("num_experts"),
        nb::arg("token_start") = 0,
        nb::arg("num_tokens") = nb::none());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
