// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "routed_expert_ffn.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail {

void bind_routed_expert_ffn(nb::module_& mod) {
    ttnn::bind_function<"routed_expert_ffn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Single expert FFN computation for DeepSeek MoE prefill.

        Computes the gated FFN for one routed expert:
            gate_out = x @ gate_proj
            up_out   = x @ up_proj
            activated = silu(gate_out) * up_out
            output   = activated @ down_proj

        Args:
            x (ttnn.Tensor): Input tensor.
            gate_proj (ttnn.Tensor): Gate projection weight (emb_dim, hidden_dim).
            up_proj (ttnn.Tensor): Up projection weight (emb_dim, hidden_dim).
            down_proj (ttnn.Tensor): Down projection weight (hidden_dim, emb_dim).

        Keyword Args:
            gate_program_config (ttnn.MatmulProgramConfig, optional): Program config for the gate matmul. Defaults to None.
            up_program_config (ttnn.MatmulProgramConfig, optional): Program config for the up matmul. Defaults to None.
            down_program_config (ttnn.MatmulProgramConfig, optional): Program config for the down matmul. Defaults to None.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration. Defaults to None.
            output (ttnn.Tensor, optional): Pre-allocated output tensor for in-place write of the final matmul result. Defaults to None.

        Returns:
            ttnn.Tensor: Output tensor with the same shape as ``x``.

        Example:
            >>> output = ttnn.experimental.deepseek_prefill.routed_expert_ffn(
                    x, gate_proj, up_proj, down_proj,
                    compute_kernel_config=compute_kernel_config)
        )doc",
        &routed_expert_ffn,
        nb::arg("x").noconvert(),
        nb::arg("gate_proj").noconvert(),
        nb::arg("up_proj").noconvert(),
        nb::arg("down_proj").noconvert(),
        nb::kw_only(),
        nb::arg("gate_program_config") = nb::none(),
        nb::arg("up_program_config") = nb::none(),
        nb::arg("down_program_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_routed_expert_ffn(::nanobind::module_& mod) { routed_expert_ffn::detail::bind_routed_expert_ffn(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
