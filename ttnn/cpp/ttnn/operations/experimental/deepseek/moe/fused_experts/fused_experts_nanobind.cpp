// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/moe/fused_experts/fused_experts.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail {

void bind_fused_experts(nb::module_& mod) {
    ttnn::bind_function<"fused_experts", "ttnn.experimental.deepseek.moe.">(
        mod,
        R"doc(
        Experimental fused routed-expert FFN for DeepSeek V4-Flash decode (sequence length 1).

        Fuses the per-expert matmul -> SwiGLU -> matmul -> weighted-accumulate loop
        into a single device operation. Expert selection/scaling is read on-device from
        ``routing_weights``: the i-th weight pair is scaled by ``routing_weights`` column i,
        so experts with zero routing weight contribute nothing (no host-side expert-id list).

        CURRENT MILESTONE: runs the gate_up matmul + SwiGLU activation + down matmul on device for
        the routing-selected experts and returns a [num_experts, 1, H] BFLOAT16 TILE tensor, where
        act = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit), [gate, up] = x @ gate_up_w[hit_ids[i]],
        and output[i] = act @ down_w[hit_ids[i]]; hit_ids are the nonzero ``routing_weights`` columns
        in ascending order. The gate_up weights must be DRAM ND-sharded so each shard is one core's
        [H, 64] slice (gate/up columns interleaved at tile granularity), and the down weights DRAM
        ND-sharded so each shard is one core's [I, H/64] slice — both read in a single NoC read. The
        SwiGLU activation is gathered onto core {0,0} and broadcast to every core for the down matmul.
        ``input_tensor`` must be TILE layout and ``routing_weights`` ROW_MAJOR bfloat16. Routing-weight
        scaling + cross-expert accumulation are later milestones.

        Args:
            input_tensor: Activations, [1, 1, 1, H].
            routing_weights: Per-token routing weights, [1, 1, 1, E] (ROW_MAJOR bfloat16),
                with E == len(gate_up_weights).
            gate_up_weights: List of [H, 2I] weight tensors, one per expert (all experts provided),
                with gate/up columns interleaved at tile (32-col) granularity.
            down_weights: List of [I, H] weight tensors, one per expert.
            num_experts: Number of routing-selected ("hit") experts to run; must equal the number of
                nonzero routing-weight columns. Also the output's leading dimension.
            intermediate_size: SwiGLU intermediate size I (output's last dimension).
            swiglu_limit: Clamp limit used by the SwiGLU activation.
            memory_config: Optional output memory config.
        )doc",
        &ttnn::experimental::deepseek::moe::fused_experts,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("routing_weights"),
        nb::arg("gate_up_weights"),
        nb::arg("down_weights"),
        nb::arg("num_experts"),
        nb::arg("intermediate_size"),
        nb::arg("swiglu_limit"),
        nb::arg("memory_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_fused_experts(::nanobind::module_& mod) { fused_experts::detail::bind_fused_experts(mod); }

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
