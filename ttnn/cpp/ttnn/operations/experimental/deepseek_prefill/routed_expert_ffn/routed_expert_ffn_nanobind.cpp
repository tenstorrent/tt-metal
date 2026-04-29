// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail {

void bind_routed_expert_ffn(nb::module_& mod) {
    nb::enum_<RoutedExpertMode>(mod, "RoutedExpertMode")
        .value("SILU", RoutedExpertMode::SILU)
        .value("GPT_OSS_SWIGLU", RoutedExpertMode::GPT_OSS_SWIGLU)
        .export_values();

    ttnn::bind_function<"routed_expert_ffn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Single expert FFN computation for routed-MoE prefill (DeepSeek + GPT-OSS).

        For mode=SILU (DeepSeek):
            gate_out  = silu(x @ gate_proj + gate_bias)
            up_out    = x @ up_proj + up_bias
            activated = gate_out * up_out
            output    = activated @ down_proj + down_bias

        For mode=GPT_OSS_SWIGLU (GPT-OSS):
            gate_out  = x @ gate_proj + gate_bias                       (no fused activation)
            up_out    = x @ up_proj + up_bias
            activated = gpt_oss_swiglu(gate_out, up_out)                (single SFPU pass:
                                                                         clamped + sigmoid)
            output    = activated @ down_proj + down_bias

        All biases are optional; pass None to skip the bias add.

        Args:
            x (ttnn.Tensor): Input tensor.
            gate_proj (ttnn.Tensor): Gate projection weight (emb_dim, hidden_dim).
            up_proj (ttnn.Tensor): Up projection weight (emb_dim, hidden_dim).
            down_proj (ttnn.Tensor): Down projection weight (hidden_dim, emb_dim).

        Keyword Args:
            mode (RoutedExpertMode): SILU (DeepSeek) or GPT_OSS_SWIGLU (GPT-OSS clamped SwiGLU).
                Defaults to SILU.
            gate_bias (ttnn.Tensor, optional): Bias added to the gate matmul output.
            up_bias (ttnn.Tensor, optional): Bias added to the up matmul output.
            down_bias (ttnn.Tensor, optional): Bias added to the down matmul output.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): Compute kernel configuration.
            output (ttnn.Tensor, optional): Pre-allocated output tensor for in-place write of the final matmul result.

        Returns:
            ttnn.Tensor: Output tensor with the same shape as ``x``.
        )doc",
        &routed_expert_ffn,
        nb::arg("x").noconvert(),
        nb::arg("gate_proj").noconvert(),
        nb::arg("up_proj").noconvert(),
        nb::arg("down_proj").noconvert(),
        nb::kw_only(),
        nb::arg("mode") = RoutedExpertMode::SILU,
        nb::arg("gate_bias") = nb::none(),
        nb::arg("up_bias") = nb::none(),
        nb::arg("down_bias") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_routed_expert_ffn(::nanobind::module_& mod) { routed_expert_ffn::detail::bind_routed_expert_ffn(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
