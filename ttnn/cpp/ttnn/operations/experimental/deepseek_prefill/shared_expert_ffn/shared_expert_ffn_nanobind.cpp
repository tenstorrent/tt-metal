// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_expert_ffn_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/core_coord.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "shared_expert_ffn.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_shared_expert_ffn(nb::module_& mod) {
    using shared_expert_ffn::RoutedExpertActivation;
    ttnn::bind_function<"shared_expert_ffn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Fused shared-expert GLU FFN (Blackhole), in one device op:
        ``act(x @ gate_proj, x @ up_proj) @ down_proj``.

        The unified routed-expert program run as one expert owning every row of x, so it needs no
        counts / index / region-offset tensors. x is read as given (TILE bf16, TILE bf8_b or
        ROW_MAJOR bf16) and the output is a fresh TILE bf8_b tensor. A down_proj holding one TP
        column's emb slice yields that column's slice of the output.

        Args:
            x (ttnn.Tensor): (1, .., 1, M, emb), DRAM-interleaved, M a multiple of 32.
            gate_proj (ttnn.Tensor): (emb, hidden).
            up_proj (ttnn.Tensor): (emb, hidden).
            down_proj (ttnn.Tensor): (hidden, N); N is emb or a TP slice of it.

        Keyword Args:
            activation (ttnn.RoutedExpertActivation, optional): Silu (default), SwiGluOai,
                SituGlu or ClampedSiluGlu.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional)
            core_grid_origin (ttnn.CoreCoord, optional): top-left logical core of the fixed
                ``UNIFIED_ROUTED_EXPERT_CORE_GRID`` rectangle. Defaults to (0, 0).

        Returns:
            ttnn.Tensor: (1, .., 1, M, N), TILE bf8_b.
        )doc",
        &shared_expert_ffn::shared_expert_ffn,
        nb::arg("x").noconvert(),
        nb::arg("gate_proj").noconvert(),
        nb::arg("up_proj").noconvert(),
        nb::arg("down_proj").noconvert(),
        nb::kw_only(),
        nb::arg("activation") = RoutedExpertActivation::Silu,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("core_grid_origin") = CoreCoord{0, 0});
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
