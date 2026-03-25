// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
            Counts occurrences of each expert index in a height-sharded input tensor (bincount / histogram),
            masked by an expert dispatch table that maps experts to chip IDs.

            Input tensor must be a 2D UINT16 height-sharded ROW_MAJOR tensor of shape [sp_dim, topk_dim]
            containing expert indices selected for each token.

            Expert dispatch table must be a UINT32 ROW_MAJOR tensor of shape [n_routed_experts] or
            [1, n_routed_experts]. Values encode chip placement: 0xFFFFFFFF (-1 as int32) means the
            expert is absent (skipped), any other value means the expert is present (counted).

            Returns a 1D UINT32 tensor of shape [n_routed_experts] where each element is the
            count of how many times the corresponding expert index appears in the input,
            or zero if that expert is masked out.

            Args:
                * :attr:`input_tensor`: 2D UINT16 height-sharded tensor of expert indices [sp_dim, topk_dim].
                * :attr:`expert_mask`: UINT32 tensor of shape [n_routed_experts] or [1, n_routed_experts] (0xFFFFFFFF = skip, other = count).
                * :attr:`n_routed_experts`: Number of routed experts (output dimension size).

        )doc",
        ttnn::overload_t(
            &masked_bincount,
            nb::arg("input_tensor").noconvert(),
            nb::arg("expert_mask").noconvert(),
            nb::arg("n_routed_experts")));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::masked_bincount::detail
