// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "moe.hpp"

namespace ttnn::operations::experimental::moe::detail {

void bind_moe(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::moe,
        R"doc(
        moe(input_tensor, w0_tensor, w1_tensor, w2_tensor, output_tensor, math_fidelity, fp32_dest_acc_en)

        Experimental, high-performance MoE operation.

        Args:
            input_tensor: Input tensor (sharded)
            w0_tensor: Weight tensor for first matmul
            w1_tensor: Weight tensor for second matmul
            w2_tensor: Weight tensor for third matmul
            output_tensor: Output tensor (sharded)
            num_experts: Number of experts per layer
            layer_id: The layer for which the MoE operation is being performed
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("w0_tensor"),
            nb::arg("w1_tensor"),
            nb::arg("w2_tensor"),
            nb::arg("output_tensor"),
            nb::arg("num_experts"),
            nb::arg("layer_id"),
        });
}

}  // namespace ttnn::operations::experimental::moe::detail
