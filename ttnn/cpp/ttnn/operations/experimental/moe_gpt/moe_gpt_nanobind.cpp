// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "moe_gpt.hpp"

namespace ttnn::operations::experimental::moe_gpt::detail {

void bind_moe_gpt(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::moe_gpt,
        R"doc(
        Experimental, high-performance MoE operation for GPT.

        Args:
            input_tensor: Input tensor (sharded)
            w0_w1_tensor: Interleaved tensors for first and second matmul
            w2_tensor: Weight tensor for third matmul
            b0_b1_tensor: Bias tensor for gate/up projections (sharded like w0_w1)
            b2_tensor: Bias tensor for down projection (sharded like w2)
            output_tensor: Output tensor (sharded)
            num_experts: Number of experts per layer
            layer_id: The layer for which the MoE operation is being performed
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("w0_w1_tensor"),
            nb::arg("w2_tensor"),
            nb::arg("b0_b1_tensor"),
            nb::arg("b2_tensor"),
            nb::arg("output_tensor"),
            nb::arg("num_experts"),
            nb::arg("layer_id"),
            nb::arg("enable_dram_output") = false,
        });
}

}  // namespace ttnn::operations::experimental::moe_gpt::detail
