// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dit_rms_norm_unary_fused_nanobind.hpp"

#include <optional>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "dit_rms_norm_unary_fused.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor dit_rms_norm_unary_fused_wrapper(
    const ttnn::Tensor& input_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& weight,
    const std::optional<ttnn::Tensor>& bias,
    const std::optional<ttnn::Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<std::string>& activation) {
    std::optional<ttnn::operations::unary::UnaryWithParam> act_param = std::nullopt;
    if (activation.has_value()) {
        act_param = ttnn::operations::unary::utils::string_to_unary_with_param(activation.value());
    }
    return ttnn::experimental::dit_rms_norm_unary_fused(
        input_tensor,
        epsilon,
        weight,
        bias,
        residual_input_tensor,
        memory_config,
        program_config,
        compute_kernel_config,
        act_param);
}

void bind_dit_rms_norm_unary_fused(nb::module_& mod) {
    const char* doc = R"doc(
        dit_rms_norm_unary_fused(input_tensor, epsilon=1e-5, weight=None, bias=None, residual_input_tensor=None, *, memory_config=None, program_config=None, compute_kernel_config=None, activation=None)

        Fused RMSNorm + unary activation for DiT transformer blocks.

        Equivalent to ``ttnn.<activation>(ttnn.rms_norm(x, ...))`` (e.g. ``ttnn.silu(...)`` or ``ttnn.gelu(...)``)
        but computed in a single kernel pass, avoiding the intermediate tensor write/read.

        Parameters
        ----------
        input_tensor : ttnn.Tensor
            Input tensor. If interleaved, then can be TILE or ROW_MAJOR layout. If sharded, then input tensor must be TILE layout.

        epsilon : float, default: 1e-5
            Small constant added to the variance for numerical stability.

        weight : Optional[ttnn.Tensor], default: None
            Optional per-channel scale (gamma). Shape: [..., 1, hidden_dim].

        bias : Optional[ttnn.Tensor], default: None
            Optional per-channel shift (beta). Shape: [..., 1, hidden_dim].

        residual_input_tensor : Optional[ttnn.Tensor], default: None
            Optional residual tensor for fused pre-add: computes RMSNorm(input + residual).

        Keyword Args
        ------------
        memory_config : Optional[ttnn.MemoryConfig], default: None
            Memory configuration for the output tensor. If not provided, inherits from input.

        program_config : Optional[LayerNormProgramConfig], default: None
            Program configuration (sharded or default). If not provided, auto-selected based on input shard spec.

        compute_kernel_config : Optional[ttnn.DeviceComputeKernelConfig], default: None
            Compute kernel configuration. If not provided, defaults to HiFi4 with approx mode.

        activation : Optional[Union[str, ttnn.UnaryOpType, ttnn.UnaryWithParam]], default: None
            Unary activation to apply after normalization. Supported forms:

            * string: ``"silu"`` or ``"gelu"``
            * ``ttnn.UnaryOpType``: e.g. ``ttnn.UnaryOpType.SILU``
            * ``ttnn.UnaryWithParam``: for activations with additional parameters

            If ``None``, no activation is applied (equivalent to plain rms_norm).

        Returns
        -------
        ttnn.Tensor
            Normalized (and optionally activated) output tensor with the same shape and layout as input.
        )doc";

    ttnn::bind_function<"dit_rms_norm_unary_fused", "ttnn.experimental.">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::experimental::dit_rms_norm_unary_fused,
            nb::arg("input_tensor"),
            nb::arg("epsilon") = 1e-5f,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("residual_input_tensor") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("activation") = nb::none()),
        ttnn::overload_t(
            &dit_rms_norm_unary_fused_wrapper,
            nb::arg("input_tensor"),
            nb::arg("epsilon") = 1e-5f,
            nb::arg("weight") = nb::none(),
            nb::arg("bias") = nb::none(),
            nb::arg("residual_input_tensor") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("activation") = nb::none()));
}

}  // namespace ttnn::operations::experimental::transformer
