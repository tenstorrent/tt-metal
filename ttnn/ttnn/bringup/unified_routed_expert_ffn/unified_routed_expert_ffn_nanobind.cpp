// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn_nanobind.hpp"

#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/core_coord.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "unified_routed_expert_ffn.hpp"

namespace ttnn::operations::bringup::unified_routed_expert_ffn::detail {

void bind_unified_routed_expert_ffn(nb::module_& mod) {
    // Activation variant for the fused routed-expert FFN. Registered before the
    // function bindings so it can serve as a default kwarg value.
    nb::enum_<RoutedExpertActivation>(mod, "RoutedExpertActivation")
        .value("Silu", RoutedExpertActivation::Silu)
        .value("SwiGluOai", RoutedExpertActivation::SwiGluOai)
        .value("SituGlu", RoutedExpertActivation::SituGlu)
        .value("ClampedSiluGlu", RoutedExpertActivation::ClampedSiluGlu)
        .value("GeluTanh", RoutedExpertActivation::GeluTanh);

    // The worker rectangle this op fixes regardless of device grid. Exported so a hybrid forward can
    // pass moe_fused_swiglu the same grid -- that op defaults to the whole device instead -- without
    // a second copy of the numbers on the Python side.
    mod.attr("UNIFIED_ROUTED_EXPERT_CORE_GRID") = CoreCoord{kCoreGridX, kCoreGridY};

    ttnn::bind_function<"unified_routed_expert_moe", "ttnn.bringup.">(
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
                Silu (default, DeepSeek), SwiGluOai (clamped, MiniMax-M3 / gpt-oss),
                SituGlu (tanh-capped, Kimi K3; Blackhole only), or ClampedSiluGlu
                (silu(min(gate,10)) * clamp(up,+/-10), DeepSeek V4; Blackhole only),
                or GeluTanh (bring-up fork).
            gate_biases, up_biases, down_biases (list[ttnn.Tensor], optional): default None.
                Per-local-expert biases (gpt-oss): all three lists together or none, one entry
                per local expert.
            min_active_tokens (int, optional): default 0.
            max_active_tokens (int, optional): default 2**32 - 1. The active-token band this op
                owns: an expert whose token count falls outside [min, max] is dropped like a zero
                count. Wide open by default; a hybrid dispatch narrows it so this op and
                moe_fused_swiglu split the experts by load over one shared counts vector.
            high_precision (bool, optional): default False. When True, every activation
                honours compute_kernel_config's math_fidelity and fp32_dest_acc_en (by
                default only GeluTanh does; the others run LoFi with a bf16 dest), and x
                (tilized in-kernel), the intermediates and the output stay bf16 instead of
                bf8_b. Needs a ROW_MAJOR bf16 dispatched_buffer; returns a TILE bf16 buffer.
            routing_weights (ttnn.Tensor, optional): default None. FLOAT32 ROW_MAJOR [rows of
                dispatched_buffer, C >= experts_per_chip]: local expert le multiplies the activation of its
                dispatched row i by routing_weights[i, le] (fp32, SFPU) before the down projection
                (DeepSeek-V4: down(r * act(gate, up))). Needs high_precision, fp32_dest_acc_en and a fused
                binary activation (SwiGluOai, SituGlu, ClampedSiluGlu).
            fp8_act_quant (bool, optional): default False. fp8 e4m3 quantize-dequantize of the activation
                (after the routing weight) per 32 hidden values with a power-of-two (ue8m0) scale
                2^ceil(log2(max(amax, 1e-4) / 448)), round to nearest even, before the down projection
                (DeepSeek act_quant(h, 32, 'ue8m0')). Same requirements as routing_weights.
            weights_transposed (bool, optional): default False. gate/up/down are given as stored by
                nn.Linear, [N, K] (gate/up [hidden, emb], down [emb, hidden]); the op reads their tiles in
                transposed order and transposes each tile in the matmul (a bfp8 [N, K] weight keeps its
                exponent groups along K).

        Each per-expert FFN picks its chunk_M_tiles / per_core_M / num_chunks at
        RUNTIME from the device-resident token count, so there is no expected-token
        argument — the work scales to each expert's actual load automatically.

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
        nb::arg("down_biases") = nb::none(),
        nb::arg("min_active_tokens") = 0,
        nb::arg("max_active_tokens") = std::numeric_limits<uint32_t>::max(),
        nb::arg("high_precision") = false,
        nb::arg("routing_weights") = nb::none(),
        nb::arg("fp8_act_quant") = false,
        nb::arg("weights_transposed") = false);
}

}  // namespace ttnn::operations::bringup::unified_routed_expert_ffn::detail

namespace ttnn::operations::bringup::detail {

void bind_unified_routed_expert_ffn(::nanobind::module_& mod) {
    unified_routed_expert_ffn::detail::bind_unified_routed_expert_ffn(mod);
}

}  // namespace ttnn::operations::bringup::detail
