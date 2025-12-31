// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moe.hpp"

namespace nb = nanobind;

namespace ttnn::operations::reduction::detail {

void bind_reduction_moe_operation(nb::module_& mod) {
    const auto* const doc =
        R"doc(
            Returns the weight of the zero-th MoE expert.

            Note:
                This is equivalent to the following PyTorch code:
                val, ind = torch.topk(input_tensor + expert_mask_tensor, k)
                return torch.sum(torch.softmax(val+topk_mask_tensor, dim=-1)*(ind==0), dim=-1)

            Args:
                * :attr:`input_tensor`: Input Tensor for moe.
                * :attr:`expert_mask_tensor`: Expert Mask Tensor for mask to out invalid experts.
                * :attr:`topk_mask_tensor`: Topk Mask Tensor for mask to out everything except topk.
                * :attr:`k`: the number of top elements to look for

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensors
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensors

            Returns:
                ttnn.Tensor: the output tensor.
        )doc";

    mod.def(
        "moe",
        &ttnn::moe,
        doc,
        nb::arg("input_tensor").noconvert(),
        nb::arg("expert_mask_tensor").noconvert(),
        nb::arg("topk_mask_tensor").noconvert(),
        nb::arg("k") = 32,
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::reduction::detail
