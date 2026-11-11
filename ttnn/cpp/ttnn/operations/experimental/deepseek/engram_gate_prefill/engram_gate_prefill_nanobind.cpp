// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "engram_gate_prefill_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/engram_gate_prefill/engram_gate_prefill.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_engram_gate_prefill(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"engram_gate_prefill", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Fused prefill Engram gate: everything ``DeepSeekV41PrefillEngram.forward`` does after the ``wkv`` matmul,
        in one program. Same math as ``engram_gate`` (the decode op), on DRAM-interleaved TILE operands::

            key, value = kv[..., :hc*D], kv[..., hc*D:]       (value broadcast over the hc copies)
            rstd = rsqrt(mean(x^2) + eps) * rsqrt(mean(key^2) + eps)
            dot  = sum(x * gate_weight * key) * rstd
            gate = sigmoid(sign(dot) * sqrt(max(|dot|, clamp_value)))
            out  = x + gate * value

        All reductions are over D, which a cohort of cores splits; the partials are summed with an all-to-all
        exchange inside the op. Replaces the slice/concat, the fp32 typecasts and the elementwise chain of the
        plain ``ttnn`` version with a single dispatch.

        Args:
            streams (ttnn.Tensor): ``[1, hc, T, D]`` bf16, TILE, DRAM interleaved. T must be a multiple of 32.
            kv (ttnn.Tensor): ``[1, 1, T, (hc+1)*D]`` bf16, TILE, DRAM interleaved -- the ``wkv`` projection.
            gate_weight (ttnn.Tensor): ``[1, hc, 32, D]`` fp32, TILE, DRAM interleaved. Page ``h`` is
                ``q_weight * k_weight / sqrt(D)`` of copy ``h``, repeated down the page's 32 rows.

        Keyword Args:
            eps (float): RMS epsilon.
            clamp_value (float): floor of ``|dot|`` before the signed square root.
            memory_config (Optional[ttnn.MemoryConfig]): output memory config; must equal ``streams``' (default).
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): defaults to HiFi4 / fp32 dest acc,
                which the op requires.

        Returns:
            ttnn.Tensor: ``[1, hc, T, D]`` with the same dtype and memory config as ``streams``.
        )doc",
        &ttnn::experimental::deepseek::engram_gate_prefill,
        nb::arg("streams"),
        nb::arg("kv"),
        nb::arg("gate_weight"),
        nb::arg("eps"),
        nb::kw_only(),
        nb::arg("clamp_value") = 1e-6f,
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
