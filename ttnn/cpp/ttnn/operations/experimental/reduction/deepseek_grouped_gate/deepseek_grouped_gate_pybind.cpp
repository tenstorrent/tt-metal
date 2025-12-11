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
            Gating mechanism for routing inputs in a mixture-of-experts (MoE) model, specifically optimized for DeepSeek.

            Args:
                scores (ttnn.Tensor): Input scores tensor.
                bias (ttnn.Tensor): Bias tensor.
                route_scale (float): Routing scale factor.
                epsilon (float): Epsilon for numerical stability.
                n_groups (int): Number of groups to partition the experts into.
                summed_experts_per_group (int): Number of experts per group to sum prior to ranking groups.
                topk_groups (int): Number of top groups to select from.
                n_activated_experts (int): Number of final experts to select per token.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to None.

            Returns:
                Tuple[ttnn.Tensor, ttnn.Tensor]: A tuple containing the scaled expert scores and selected expert indices.
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
