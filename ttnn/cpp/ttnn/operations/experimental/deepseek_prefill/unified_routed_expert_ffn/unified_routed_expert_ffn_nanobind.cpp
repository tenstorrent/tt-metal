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
    ttnn::bind_function<"unified_routed_expert_ffn", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Single-op fused routed-expert FFN for DeepSeek V3 prefill (Blackhole).

        Computes the entire SwiGLU FFN sequence in ONE device program:
            gate = matmul(x, gate_proj)
            up   = matmul(x, up_proj)
            y    = matmul(silu(gate) * up, down_proj)

        The kernel reads the device-resident token count
        ``counts[global_expert_idx_table[local_expert_id]]`` at runtime and
        skips M-chunks beyond that count.

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
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("local_expert_id"),
        nb::arg("use_region_offsets") = true,
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output") = nb::none());

    ttnn::bind_function<"unified_routed_expert_moe", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        MoE-level composite op: takes the full dispatched buffer + ALL local
        experts' weights and loops per-expert internally, calling
            extract -> unified_routed_expert_ffn -> insert.

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
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_unified_routed_expert_ffn(::nanobind::module_& mod) {
    unified_routed_expert_ffn::detail::bind_unified_routed_expert_ffn(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
