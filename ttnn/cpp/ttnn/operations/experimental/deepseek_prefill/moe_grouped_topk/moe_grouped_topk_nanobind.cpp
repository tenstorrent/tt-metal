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
            Gating mechanism for routing inputs in a mixture-of-experts (MoE) model, specifically
            optimized for DeepSeek. It post-processes the scores and bias from the linear gate
            projection through the following stages:

              1. Activation: apply the router affinity activation (score_func) to each score.
              2. Bias: add the bias and reshape the biased scores into ``n_groups`` groups.
              3. Group ranking: sort scores within each group, sum the top ``summed_experts_per_group``
                 experts per group, and select the top ``topk_groups`` groups by that sum.
              4. Expert selection: select the top ``n_activated_experts`` experts from the selected
                 groups, then gather the unbiased scores (the activation output, without the bias)
                 for those experts.
              5. Normalize and scale: normalize the gathered scores and scale them by ``route_scale``.

            Args:
                scores (ttnn.Tensor): Input scores tensor (dtype must be FLOAT32 or BFLOAT16, layout must be TILE). BFLOAT16 inputs are upcast to FLOAT32 inside the kernel, so the op computes in fp32 either way (this avoids a separate host-side typecast). The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                bias (ttnn.Tensor): Bias tensor (dtype must be FLOAT32 or BFLOAT16, layout must be TILE). BFLOAT16 inputs are upcast to FLOAT32 inside the kernel. The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                n_groups (int): Number of groups to partition the experts into. Right now this number must be 8.
                summed_experts_per_group (int): Number of experts per group to sum prior to ranking groups. Right now this number must be 2.
                topk_groups (int): Number of top groups to select from. Right now this number must be 4.
                n_activated_experts (int): Number of final experts to select per token. Right now this number must be 8.
                route_scale (float): Routing scale factor to scale the scores after normalization.
                epsilon (float): Epsilon for numerical stability when normalizing the scores.
                stable_sort (bool): Use stable sorting in topk to maintain relative order of equal-valued elements. Defaults to False.
                score_func (str): Router affinity activation applied to the logits. "sigmoid" (DeepSeek-V3 / Kimi, default) or "sqrtsoftplus" (DeepSeek-V4, sqrt(softplus(x))).
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to None, which results in auto-selection.
                padding_config (ttnn.Tensor, optional): ROW_MAJOR UINT32 tensor with per-device [num_real_tokens, pad_side].
                    pad_side is 0 for right padding and 1 for left padding. Defaults to None, which treats all tokens as real.

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
        nb::arg("score_func") = "sigmoid",
        nb::arg("memory_config") = nb::none(),
        nb::arg("padding_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::detail
