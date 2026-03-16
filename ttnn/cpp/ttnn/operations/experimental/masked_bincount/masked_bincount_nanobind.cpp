// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/masked_bincount/masked_bincount.hpp"

namespace ttnn::operations::experimental::masked_bincount::detail {
void bind_experimental_masked_bincount_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Counts occurrences of each expert index in a height-sharded input tensor (bincount / histogram),
            masked by a per-expert presence vector.

            Input tensor must be a 2D UINT16 height-sharded ROW_MAJOR tensor of shape [sp_dim, topk_dim]
            containing expert indices selected for each token.

            Expert mask must be a 1D UINT32 ROW_MAJOR tensor of shape [n_routed_experts] where non-zero
            means the expert is present (counted) and zero means it is absent (skipped).

            Returns a 1D UINT32 tensor of shape [n_routed_experts] where each element is the
            count of how many times the corresponding expert index appears in the input,
            or zero if that expert is masked out.

            Args:
                * :attr:`input_tensor`: 2D UINT16 height-sharded tensor of expert indices [sp_dim, topk_dim].
                * :attr:`expert_mask`: 1D UINT32 tensor of shape [n_routed_experts] (0 = skip, nonzero = count).
                * :attr:`n_routed_experts`: Number of routed experts (output dimension size).

        )doc";

    using OperationType = decltype(ttnn::masked_bincount);
    bind_registered_operation(
        mod,
        ttnn::masked_bincount,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_mask,
               uint32_t n_routed_experts) { return self(input_tensor, expert_mask, n_routed_experts); },
            nb::arg("input_tensor").noconvert(),
            nb::arg("expert_mask").noconvert(),
            nb::arg("n_routed_experts")});
}

}  // namespace ttnn::operations::experimental::masked_bincount::detail
