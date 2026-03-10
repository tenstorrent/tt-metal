// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_dispatch_offsets_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/moe_dispatch_offsets/moe_dispatch_offsets.hpp"

namespace ttnn::operations::experimental::moe_dispatch_offsets::detail {
void bind_experimental_moe_dispatch_offsets_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Counts occurrences of each expert index in the input tensor (bincount / histogram).

            Input tensor must be a 2D UINT16 or UINT32, ROW_MAJOR tensor of shape [sp_dim, topk_dim]
            containing expert indices selected for each token.

            Returns a 1D UINT32 tensor of shape [n_routed_experts] where each element is the
            count of how many times the corresponding expert index appears in the input.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.bincount(input_tensor.flatten(), minlength=n_routed_experts)

            Args:
                * :attr:`input_tensor`: 2D UINT16 or UINT32 tensor of expert indices [sp_dim, topk_dim].
                * :attr:`n_routed_experts`: Number of routed experts (output dimension size).

        )doc";

    using OperationType = decltype(ttnn::moe_dispatch_offsets);
    bind_registered_operation(
        mod,
        ttnn::moe_dispatch_offsets,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self, const ttnn::Tensor& input_tensor, uint32_t n_routed_experts) {
                return self(input_tensor, n_routed_experts);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("n_routed_experts")});
}

}  // namespace ttnn::operations::experimental::moe_dispatch_offsets::detail
