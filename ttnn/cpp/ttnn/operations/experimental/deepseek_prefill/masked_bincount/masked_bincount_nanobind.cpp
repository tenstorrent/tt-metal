// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "masked_bincount.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount::detail {
void bind_experimental_masked_bincount_operation(nb::module_& mod) {
    ttnn::bind_function<"masked_bincount", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Counts occurrences of each expert in the input tensor. Only the experts in the current
            dispatch group are taken into account. Expert dispatch table (expert_mask) maps experts
            to chip IDs, where negative values indicate masked-out experts (experts from different
            dispatch groups).

            Args:
                * :attr:`input_tensor`: 2D UINT16 TILE, interleaved tensor of shape [sp_dim, topk_dim]
                  containing the expert indices selected for each token. This is the gate's expert-index
                  output consumed directly (no untilize/reshard needed): the op untiles in-kernel and
                  splits the token rows across a fixed 8x8 (64-core) grid internally. sp_dim must be a
                  multiple of 64. Both DRAM- and L1-interleaved inputs are accepted (the gate emits L1).
                * :attr:`expert_mask`: INT32 ROW_MAJOR tensor of shape [N] or [1, N] where N >= n_routed_experts,
                  mapping experts to chip IDs. Negative (-1) means the expert is absent (belongs to different dispatch
                  groups); non-negative values (chip IDs) mean the expert is present in this dispatch group and will
                  be counted. A sentinel column at index == n_routed_experts is allowed and ignored: the kernel only
                  reads mask entries at index < n_routed_experts, so padded tokens (sentinel index == n_routed_experts)
                  are skipped and never counted.
                * :attr:`n_routed_experts`: Number of routed experts (output dimension size).
                * :attr:`num_experts_per_token`: Number of expert columns per row to count (must be <= the logical
                  topk_dim). Columns beyond this index (including the TILE width padding up to 32) are ignored.

            Returns:
                A 1D UINT32 tensor of shape [n_routed_experts] where each element is the count of how many
                times the corresponding expert index appears in the input, or zero if that expert is masked out.

        )doc",
        &masked_bincount,
        nb::arg("input_tensor").noconvert(),
        nb::arg("expert_mask").noconvert(),
        nb::arg("n_routed_experts"),
        nb::arg("num_experts_per_token"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount::detail
