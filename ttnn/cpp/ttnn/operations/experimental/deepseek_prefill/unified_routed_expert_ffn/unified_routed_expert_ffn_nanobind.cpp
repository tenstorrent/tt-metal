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

    ttnn::bind_function<"unified_routed_expert_ffn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Single-op fused per-expert FFN for DeepSeek V3 prefill (Blackhole).

        Computes the entire SwiGLU FFN sequence in ONE device program:
            gate = matmul(x, gate_proj)
            up   = matmul(x, up_proj)
            y    = matmul(silu(gate) * up, down_proj)

        The kernel reads the device-resident token count
        ``counts[global_expert_idx_table[local_expert_id]]`` at runtime and
        skips M-chunks beyond that count. x is the already-extracted per-expert
        tokens tensor (rows start at 0); use ``unified_routed_expert_moe`` below
        if you need the extract/insert glue.

        Tensor requirements (enforced in validate_on_program_cache_miss):
            * dtype: x must be BFLOAT8_B (BFLOAT16 path is untested and
              rejected host-side; reintroduce when a real caller + PCC test
              lands); gate/up/down any matmul-compatible weight dtype.
            * layout: all tensors TILE.
            * memory_config: all tensors DRAM-interleaved.
            * Blackhole-only — host expects 11x8 compute grid.

        PCC target: >= 0.97 vs PyTorch reference (matches the sibling
        routed_expert_ffn subsystem norm; the existing test_unified_routed_expert
        cases land at ~0.98 on DS-V3 dims with LoFi math fidelity).

        Args:
            x (ttnn.Tensor): (M_max, K=emb) DRAM-interleaved tile-layout input.
            gate_proj (ttnn.Tensor): (K=emb, N=hidden).
            up_proj (ttnn.Tensor): (K=emb, N=hidden).
            down_proj (ttnn.Tensor): (K=hidden, N=emb).
            counts (ttnn.Tensor): UINT32, per-global-expert token counts.
            global_expert_idx_table (ttnn.Tensor): UINT32, maps local id -> global.
            local_expert_id (int): index into global_expert_idx_table.

        Keyword Args:
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional)
            output (ttnn.Tensor, optional): pre-allocated output buffer.
                Must match x.dtype() and x.shape() unless expert_region_offsets
                is set (direct-write mode), in which case it is the larger
                shared destination buffer.
            expert_region_offsets (ttnn.Tensor, optional): UINT32
                per-global-expert region start offsets. When set, the writer
                places this expert's output directly into ``output`` at
                start[global_id]/TILE tile-rows (direct-write mode), fusing the
                ttnn::insert step. Requires ``output`` to be set. Defaults to
                None (standalone per-expert output, rows start at 0).
            input_m_tiles (int, optional): this expert's M in tiles. Defaults to
                x's allocated M. Supply it when x is a shared buffer wider than
                one expert's region so the op sizes its work to this expert.
            read_x_at_offset (bool, optional): when True, x is a shared buffer
                and the reader offsets x reads by expert_region_offsets[global_id]
                (fusing ttnn::extract). Requires expert_region_offsets. Default False.
            activation (ttnn.RoutedExpertActivation, optional):
                Silu (default, DeepSeek) or SwiGluOai (clamped, MiniMax-M3 / gpt-oss).

        Returns:
            ttnn.Tensor: (M_max, K=emb).
        )doc",
        &unified_routed_expert_ffn,
        nb::arg("x").noconvert(),
        nb::arg("gate_proj").noconvert(),
        nb::arg("up_proj").noconvert(),
        nb::arg("down_proj").noconvert(),
        nb::arg("counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::arg("local_expert_id"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output") = nb::none(),
        nb::arg("expert_region_offsets") = nb::none(),
        nb::arg("input_m_tiles") = nb::none(),
        nb::arg("read_x_at_offset") = false,
        nb::arg("activation") = RoutedExpertActivation::Silu);

    ttnn::bind_function<"unified_routed_expert_moe", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        MoE-level composite: takes the full dispatched buffer + ALL local
        experts' weights and loops over local experts in C++, launching one
        ``unified_routed_expert_ffn`` device program per expert preceded by
        ``ttnn::extract`` (input slice). The FFN runs in direct-write mode:
        its writer places each expert's output straight into the shared
        output buffer at the expert's region offset, so NO separate
        ``ttnn::insert`` op (and no per-expert temp-buffer DRAM round-trip)
        is needed. This is NOT a single fused device op across experts —
        per-expert FFN entries still appear in tt-perf-report.

        The unified FFN reads device-resident counts/idx and bounds its
        chunk loop to the actually-occupied chunks per expert. The host
        does NOT sync to read counts.

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
        nb::arg("activation") = RoutedExpertActivation::Silu);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_unified_routed_expert_ffn(::nanobind::module_& mod) {
    unified_routed_expert_ffn::detail::bind_unified_routed_expert_ffn(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
