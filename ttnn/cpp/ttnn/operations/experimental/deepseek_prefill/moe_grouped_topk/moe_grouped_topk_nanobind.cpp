// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_grouped_topk_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/moe_grouped_topk.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::detail {

void bind_moe_grouped_topk(nb::module_& mod) {
    ttnn::bind_function<"moe_grouped_topk", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Gating mechanism for routing inputs in a mixture-of-experts (MoE) model, specifically optimized for DeepSeek. This operation post-processes the scores and bias from the linear gate projection. To each score, it applies a sigmoid, adds the bias, and reshapes the scores into groups. It then sorts the scores within each group, sums the scores of the top p experts in each group, and selects the top k groups. It then selects the top n experts from the selected groups and gathers the unbiased scores (original sigmoid output) based on the top k experts indices. It then normalizes the chosen scores and scales the normalized scores by the scales.

            Args:
                scores (ttnn.Tensor): Input scores tensor (dtype must be FLOAT32, layout must be TILE). The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                bias (ttnn.Tensor): Bias tensor (dtype must be FLOAT32, layout must be TILE). The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                n_groups (int): Number of groups to partition the experts into. Right now this number must be 8.
                summed_experts_per_group (int): Number of experts per group to sum prior to ranking groups. Right now this number must be 2.
                topk_groups (int): Number of top groups to select from. Right now this number must be 4.
                n_activated_experts (int): Number of final experts to select per token. Right now this number must be 8.
                route_scale (float): Routing scale factor to scale the scores after normalization.
                epsilon (float): Epsilon for numerical stability when normalizing the scores.
                stable_sort (bool): Use stable sorting in topk to maintain relative order of equal-valued elements. Defaults to False.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to None, which results in auto-selection.

            Returns:
                Tuple[ttnn.Tensor, ttnn.Tensor]: A tuple containing the scaled expert scores (dtype BFLOAT16) and selected expert indices (dtype UINT16). The shape of the scores tensor should be [N, B, S, 8]. The shape of the indices tensor should be [N, B, S, 8]. N, B and S can be any value. 8 is the number of experts in the final selected groups.
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::moe_grouped_topk,
        nb::arg("scores").noconvert(),
        nb::arg("bias").noconvert(),
        nb::kw_only(),
        nb::arg("n_groups") = 8,
        nb::arg("summed_experts_per_group") = 2,
        nb::arg("topk_groups") = 4,
        nb::arg("n_activated_experts") = 8,
        nb::arg("route_scale") = 1.0f,
        nb::arg("epsilon") = 1e-20f,
        nb::arg("stable_sort") = false,
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::detail
