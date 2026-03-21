// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include "fused_persistent_moe_decode.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::fused_persistent_moe_decode {

void bind_fused_persistent_moe_decode(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::fused_persistent_moe_decode,
        R"doc(
        Fused Persistent MoE Decode Kernel.
        Computes local Top-4 router and overlaps compute with async NOC prefetch.
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::fused_persistent_moe_decode)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& topk_expert_indices,
               const ttnn::Tensor& topk_expert_weights,
               const ttnn::Tensor& w1_experts,
               const ttnn::Tensor& w3_experts,
               const ttnn::Tensor& w2_experts) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    topk_expert_indices,
                    topk_expert_weights,
                    w1_experts,
                    w3_experts,
                    w2_experts);
            },
            nb::arg("input_tensor"),
            nb::arg("topk_expert_indices"),
            nb::arg("topk_expert_weights"),
            nb::arg("w1_experts"),
            nb::arg("w3_experts"),
            nb::arg("w2_experts")
        }
    );
}

} // namespace ttnn::operations::experimental::fused_persistent_moe_decode
