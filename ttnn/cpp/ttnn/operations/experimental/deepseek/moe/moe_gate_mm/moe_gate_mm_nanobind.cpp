// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gate_mm_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "moe_gate_mm.hpp"

namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::detail {

void bind_moe_gate_mm(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::deepseek::moe::moe_gate_mm,
        R"doc(
        Experimental, high-performance MoE Gate MM operation for DeepSeek.

        Args:
            input_tensor: Input tensor (sharded)
            w_tensor: Weight tensor
            output_tensor: Output tensor (sharded)
            layer_id: The layer for which the MoE Gate MM operation is being performed
            column_id: The column in the 16x8 grid in which this MoE gate is being performed
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("w_tensor"),
            nb::arg("output_tensor"),
            nb::arg("layer_id"),
            nb::arg("column_id"),
        });
}

}  // namespace ttnn::operations::experimental::deepseek::moe::moe_gate_mm::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_moe_gate_mm(::nanobind::module_& mod) { moe_gate_mm::detail::bind_moe_gate_mm(mod); }

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
