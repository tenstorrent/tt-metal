// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_combine_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "routed_combine.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

void bind_routed_combine(nb::module_& mod) {
    ttnn::bind_function<"routed_combine", "ttnn.experimental.nomic_embed_text_v2_moe.">(
        mod,
        R"doc(
        Combines a routed MoE pass's expert outputs back into one row per token.

        For each token t of the window, out[t] = sum over k of combine_weights[k, t] *
        expert_rows[combine_index[k, t]], plus bias. The weighted sum and the bias accumulate in fp32
        and round to bf16 once. Padding tokens carry weight 0 on a real row, so the output's tile
        padding holds exactly the bias. Nothing is read back to the host, so the call is trace-safe.

        Without combine_weights and bias it gathers: top_k 1, out[t] = expert_rows[combine_index[0, t]],
        tilized with no arithmetic, bit for bit what ttnn.embedding returns. Their shared pack turns
        -0.0 into 0.0, NaN into inf and denormals into 0.

        Args:
            expert_rows (ttnn.Tensor): (1, 1, rows, hidden) BFLOAT16, ROW_MAJOR, interleaved: the
                rows combine_index picks from, such as the routed experts' output rows as
                grouped_matmul writes them. A combine takes at least one row per slot, rows >= top_k *
                num_tokens rounded up to 32; a gather any count.
            combine_index (ttnn.Tensor): (top_k, padded_tokens) UINT32, ROW_MAJOR, interleaved, from
                routing_metadata. top_k is 1 to 16. Every entry must be a row of expert_rows; the
                kernel does not check them.
            combine_weights (ttnn.Tensor, optional): (1, top_k, padded_tokens, 1) FLOAT32, TILE,
                interleaved, from routing_metadata. Given with bias, or neither for a gather.
            bias (ttnn.Tensor, optional): (1, 1, 1, hidden) BFLOAT16, TILE, interleaved: added once per
                token.

        Keyword Args:
            num_tokens (int): tokens in the window; padded_tokens must be num_tokens rounded up to 32.
            memory_config (ttnn.MemoryConfig, optional): interleaved or sharded, DRAM or L1; the writer
                addresses tiles by page id. Defaults to expert_rows'.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): fp32_dest_acc_en
                must be on. Defaults to HiFi4 with fp32 accumulation.
            batch (int, optional): write the output as (batch, 1, num_tokens / batch, hidden), page for page
                the flat one; num_tokens / batch must be a multiple of 32 when batch > 1. Defaults to 1.

        Returns:
            ttnn.Tensor: (batch, 1, num_tokens / batch, hidden) BFLOAT16, TILE, its tile padding the bias, or a
            gather's rows.
        )doc",
        &ttnn::experimental::nomic_embed_text_v2_moe::routed_combine,
        nb::arg("expert_rows"),
        nb::arg("combine_index"),
        nb::arg("combine_weights") = nb::none(),
        nb::arg("bias") = nb::none(),
        nb::kw_only(),
        nb::arg("num_tokens"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("batch") = 1);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
