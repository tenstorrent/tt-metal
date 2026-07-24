// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "unified_routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail {

void bind_unified_routed_expert_ffn(nb::module_& mod) {
    // Activation variant for the fused routed-expert FFN. Registered before the
    // function bindings so it can serve as a default kwarg value.
    nb::enum_<RoutedExpertActivation>(mod, "RoutedExpertActivation")
        .value("Silu", RoutedExpertActivation::Silu)
        .value("SwiGluOai", RoutedExpertActivation::SwiGluOai);

    ttnn::bind_function<"unified_routed_expert_moe", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Single-op fused routed-expert MoE FFN (Blackhole). Takes the dispatched
        buffer + ALL local experts' weights and runs in ONE device program. The
        reader/compute/writer kernels loop over the local experts: each expert
        reads its slice of the shared dispatched buffer at its region offset
        (fusing ``ttnn::extract``), runs gate/up/down, and writes its output
        straight into the shared output buffer at that offset (fusing
        ``ttnn::insert``). NO per-expert temp buffer, no per-expert dispatch, and
        no inter-expert dispatch gap.

        The kernels read device-resident counts/idx and bound each expert's
        chunk loop to the actually-occupied chunks. The host does NOT sync to
        read counts.

        Args:
            dispatched_buffer (ttnn.Tensor): (max_dispatch, emb).
            expert_region_offsets (ttnn.Tensor): UINT32 1D, per-expert start.
            expert_token_counts (ttnn.Tensor): UINT32 1D, per-expert counts.
            global_expert_idx_table (ttnn.Tensor): UINT32 1D, local->global id map.
            gate_projs (list[ttnn.Tensor]): one (emb, hidden) per local expert.
            up_projs (list[ttnn.Tensor]): one (emb, hidden) per local expert.
            down_projs (list[ttnn.Tensor]): one (hidden, emb) per local expert.
            max_dispatched_tokens_per_expert (int): rows extract gives back.

        Keyword Args:
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional)
            activation (ttnn.RoutedExpertActivation, optional):
                Silu (default, DeepSeek) or SwiGluOai (clamped, MiniMax-M3 / gpt-oss).

        Returns:
            ttnn.Tensor: expert outputs, same shape as dispatched_buffer.
        )doc",
        &unified_routed_expert_moe,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::arg("gate_projs").noconvert(),
        nb::arg("up_projs").noconvert(),
        nb::arg("down_projs").noconvert(),
        nb::arg("max_dispatched_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("activation") = RoutedExpertActivation::Silu,
        nb::arg("gate_biases") = nb::none(),
        nb::arg("up_biases") = nb::none(),
        nb::arg("down_biases") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_unified_routed_expert_ffn(::nanobind::module_& mod) {
    unified_routed_expert_ffn::detail::bind_unified_routed_expert_ffn(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
