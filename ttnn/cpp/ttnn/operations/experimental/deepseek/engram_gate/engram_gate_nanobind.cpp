// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "engram_gate_nanobind.hpp"

#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/engram_gate/engram_gate.hpp"

namespace ttnn::operations::experimental::deepseek::detail {

void bind_engram_gate(::nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"engram_gate", "ttnn.experimental.deepseek.">(
        mod,
        R"doc(
        Fused Engram gate (deepseek_v41_flash ``DeepSeekV41Engram.forward`` after the wkv matmul)::

            key, value = kv[..., :hc*D], kv[..., hc*D:]       (value broadcast over the hc copies)
            rstd = rsqrt(mean(x^2) + eps) * rsqrt(mean(key^2) + eps)
            dot  = sum(x * gate_weight * key) * rstd
            gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
            out  = x + gate * value

        All reductions are over D, which is split across the cores of ``streams``; the partials are
        summed with an all-to-all exchange inside the op.

        Args:
            streams (ttnn.Tensor): ``[B, 1, hc, D]`` bf16, ROW_MAJOR L1 WIDTH_SHARDED over D
                (shard ``[B*hc, D / num_cores]``, at most 32 cores).
            kv (ttnn.Tensor): ``[1, 1, B, (hc+1)*D]`` bf16, ROW_MAJOR L1 WIDTH_SHARDED on any grid.
            gate_weight (ttnn.Tensor): ``[1, 1, 32, D]`` fp32, TILE L1 WIDTH_SHARDED on the
                ``streams`` grid; row ``i`` is ``q_weight * k_weight / sqrt(D)`` of copy ``i % hc``.
            eps (float): RMS epsilon.

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): output memory config; must equal the
                ``streams`` memory config (the default).
            compute_kernel_config (Optional[ttnn.DeviceComputeKernelConfig]): defaults to
                HiFi4 / fp32 dest acc, which the op requires.

        Returns:
            ttnn.Tensor: ``[B, 1, hc, D]`` with the same dtype and memory config as ``streams``.
        )doc",
        &ttnn::experimental::deepseek::engram_gate,
        nb::arg("streams"),
        nb::arg("kv"),
        nb::arg("gate_weight"),
        nb::arg("eps"),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
