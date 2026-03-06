// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_nanobind.hpp"

#include <nanobind/stl/optional.h>

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
            output_tensor: Output tensor (sharded)
            num_experts: Number of experts per layer
            layer_id: The layer for which the MoE operation is being performed
            enable_dram_output: Write matmul output to DRAM (default: false)
            sparse_buffer: Optional sparse token buffer for tilize phase
            expert_indices: Optional expert index tensor for tilize phase
            expert_scores: Optional expert score tensor for tilize phase
            expert_mapping: Optional expert-to-device mapping for tilize phase
            tilize_output: Optional pre-allocated DRAM tensor for tilize output
            cluster_axis: Optional cluster axis for multi-device dispatch
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("w0_w1_tensor"),
            nb::arg("w2_tensor"),
            nb::arg("output_tensor"),
            nb::arg("num_experts"),
            nb::arg("layer_id"),
            nb::arg("enable_dram_output") = false,
            nb::arg("sparse_buffer") = nb::none(),
            nb::arg("expert_indices") = nb::none(),
            nb::arg("expert_scores") = nb::none(),
            nb::arg("expert_mapping") = nb::none(),
            nb::arg("tilize_output") = nb::none(),
            nb::arg("cluster_axis") = nb::none(),
        });
}

}  // namespace ttnn::operations::experimental::moe_gpt::detail
