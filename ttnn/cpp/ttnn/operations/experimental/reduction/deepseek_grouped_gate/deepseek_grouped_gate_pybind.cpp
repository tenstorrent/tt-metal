// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/deepseek_grouped_gate/deepseek_grouped_gate_pybind.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/reduction/deepseek_grouped_gate/deepseek_grouped_gate.hpp"

namespace ttnn::operations::experimental::reduction::detail {

namespace py = pybind11;

void bind_deepseek_grouped_gate(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::deepseek_grouped_gate,
        R"doc(
            Gating mechanism for routing inputs in a mixture-of-experts (MoE) model, specifically optimized for DeepSeek. This operation post-processes the scores and bias from the linear gate projection. To each score, it applies a sigmoid, adds the bias, and reshapes the scores into groups. It then sorts the scores within each group, sums the scores of the top p experts in each group, and selects the top k groups. It then selects the top k experts from the selected groups and gathers the unbiased scores (original sigmoid output) based on the top k experts indices. It then normalizes the chosen scores and scales the normalized scores by the scales.

            Args:
                scores (ttnn.Tensor): Input scores tensor. The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                bias (ttnn.Tensor): Bias tensor. The shape should be [N, B, S, 256]. N, B and S can be any value. 256 is the number of experts in DeepSeek at each layer.
                route_scale (float): Routing scale factor to scale the scores after normalization.
                epsilon (float): Epsilon for numerical stability when normalizing the scores.
                n_groups (int): Number of groups to partition the experts into. Right now this number must be 8.
                summed_experts_per_group (int): Number of experts per group to sum prior to ranking groups. Right now this number must be 2.
                topk_groups (int): Number of top groups to select from. Right now this number must be 4.
                n_activated_experts (int): Number of final experts to select per token. Right now this number must be 8.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to None.

            Returns:
                Tuple[ttnn.Tensor, ttnn.Tensor]: A tuple containing the scaled expert scores and selected expert indices. The shape of the scores tensor should be [N, B, S, 8]. The shape of the indices tensor should be [N, B, S, 8]. N, B and S can be any value. 8 is the number of experts in the final selected groups.
        )doc",
        ttnn::pybind_overload_t{
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
            py::arg("scores").noconvert(),
            py::arg("bias").noconvert(),
            py::kw_only(),
            py::arg("n_groups").noconvert(),
            py::arg("summed_experts_per_group").noconvert(),
            py::arg("topk_groups").noconvert(),
            py::arg("n_activated_experts").noconvert(),
            py::arg("route_scale").noconvert() = 1.0f,
            py::arg("epsilon").noconvert() = 1e-20f,
            py::arg("memory_config").noconvert() = std::nullopt});
}

}  // namespace ttnn::operations::experimental::reduction::detail
