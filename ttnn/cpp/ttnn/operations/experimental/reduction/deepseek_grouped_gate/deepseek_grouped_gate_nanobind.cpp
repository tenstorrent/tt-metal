// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_grouped_gate_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_grouped_gate/deepseek_grouped_gate.hpp"

namespace ttnn::operations::experimental::reduction::detail {

void bind_deepseek_grouped_gate(nb::module_& mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::deepseek_grouped_gate,
        R"doc(
            Gating mechanism for routing inputs in a mixture-of-experts (MoE) model, specifically optimized for DeepSeek. This operation post-processes the scores and bias from the linear gate projection. To each score, it applies a sigmoid, adds the bias, and reshapes the scores into groups. It then sorts the scores within each group, sums the scores of the top p experts in each group, and selects the top k groups. It then selects the top n experts from the selected groups and gathers the unbiased scores (original sigmoid output) based on the top k experts indices. It then normalizes the chosen scores and scales the normalized scores by the scales.

            Args:
                scores (ttnn.Tensor): Input scores tensor. The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                bias (ttnn.Tensor): Bias tensor. The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                route_scale (float): Routing scale factor to scale the scores after normalization.
                epsilon (float): Epsilon for numerical stability when normalizing the scores.
                n_groups (int): Number of groups to partition the experts into. Right now this number must be 8.
                summed_experts_per_group (int): Number of experts per group to sum prior to ranking groups. Right now this number must be 2.
                topk_groups (int): Number of top groups to select from. Right now this number must be 4.
                n_activated_experts (int): Number of final experts to select per token. Right now this number must be 8.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to None, which results in auto-selection.

            Returns:
                Tuple[ttnn.Tensor, ttnn.Tensor]: A tuple containing the scaled expert scores and selected expert indices. The shape of the scores tensor should be [N, B, S, 8]. The shape of the indices tensor should be [N, B, S, 8]. N, B and S can be any value. 8 is the number of experts in the final selected groups.
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::deepseek_grouped_gate)& self,
               const ttnn::Tensor& scores,
               const ttnn::Tensor& bias,
               uint32_t n_groups,
               uint32_t summed_experts_per_group,
               uint32_t topk_groups,
               uint32_t n_activated_experts,
               float route_scale,
               float epsilon,
               const std::optional<ttnn::MemoryConfig>& memory_config) {
                return self(
                    scores,
                    bias,
                    n_groups,
                    summed_experts_per_group,
                    topk_groups,
                    n_activated_experts,
                    route_scale,
                    epsilon,
                    memory_config);
            },
            nb::arg("scores").noconvert(),
            nb::arg("bias").noconvert(),
            nb::kw_only(),
            nb::arg("n_groups") = 8,
            nb::arg("summed_experts_per_group") = 2,
            nb::arg("topk_groups") = 4,
            nb::arg("n_activated_experts") = 8,
            nb::arg("route_scale") = 1.0f,
            nb::arg("epsilon") = 1e-20f,
            nb::arg("memory_config") = nb::none()});
}

}  // namespace ttnn::operations::experimental::reduction::detail
