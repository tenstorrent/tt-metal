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
                * :attr:`input_tensor`: 2D UINT16 height-sharded ROW_MAJOR tensor of shape [sp_dim, topk_dim]
                  containing expert indices selected for each token.
                * :attr:`expert_mask`: INT32 ROW_MAJOR tensor of shape [n_routed_experts] or [1, n_routed_experts]
                  mapping experts to chip IDs. Negative (-1) means the expert is absent (belong to different dispatch
                  groups); non-negative values (chip IDs) mean the expert is present in this dispatch groupand will
                  be counted.
                * :attr:`n_routed_experts`: Number of routed experts (output dimension size).
                * :attr:`num_experts_per_token`: Number of expert columns per row to count (must be <= topk_dim).
                  Columns beyond this index are ignored, allowing padded shard widths.

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
