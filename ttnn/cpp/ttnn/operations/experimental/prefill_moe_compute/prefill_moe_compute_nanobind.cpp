// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_moe_compute_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "prefill_moe_compute.hpp"

namespace ttnn::operations::experimental::prefill_moe_compute::detail {

void bind_prefill_moe_compute(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::prefill_moe_compute,
        R"doc(
        Fused prefill MoE expert compute pipeline.

        Performs dispatch (hidden_states -> pkt_buf), expert compute (gate_up matmul +
        SwiGLU + down matmul), and combine (weighted accumulation) for multiple experts
        in a single program launch.

        When enable_fabric_dispatch=True, uses TT-Fabric to exchange tokens between
        devices before compute. Requires hidden_states_rm (ROW_MAJOR), staging_buf,
        and dispatch_metadata.

        Args:
            hidden_states: [1,1,P,D] BF16 TILE_LAYOUT input activation tensor
            gate_up_weights: List of per-expert [1,1,D,D_FF] BFP4_b weight tensors
            down_weights: List of per-expert [1,1,D_FF/2,D] BFP4_b weight tensors
            pkt_buf: [1,1,E*M,D] BF16 scratch buffer for dispatch
            inter_buf: [1,1,M,D_FF/2] BF16 scratch buffer for intermediate
            out_bufs: List of per-expert [1,1,P_out,D] BF16 output scratch buffers
            output: [1,1,P_out,D] BF16 pre-allocated output tensor (zero-filled)
            combine_metadata: Per-device packed routing args for combine kernel
            num_experts: Number of experts (1-4)
            num_cores: Number of matmul cores
            grid_x: Matmul core grid X dimension
            grid_y: Matmul core grid Y dimension
            reduce_recv_buf: Optional [1,1,P,D] BF16 receive buffer for fabric all-reduce
            enable_fabric_reduce: Enable cross-device fabric all-reduce of partial outputs
            hidden_states_rm: Optional [1,1,P,D] BF16 ROW_MAJOR for fabric dispatch
            staging_buf: Optional [1,1,P,D] BF16 ROW_MAJOR fabric receive buffer
            enable_fabric_dispatch: Enable fabric token dispatch (default False)
            dispatch_metadata: Per-device dispatch routing metadata
            dispatch_target_cols: Per-device target column for dispatch exchange
            per_expert_dispatch_sources: Per-device per-expert token sources
            multi_dest_dispatch_metadata: Per-device multi-destination dispatch metadata
            enable_fpu_combine: Enable FPU combine on compute cores
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("hidden_states").noconvert(),
            nb::kw_only(),
            nb::arg("gate_up_weights"),
            nb::arg("down_weights"),
            nb::arg("pkt_buf").noconvert(),
            nb::arg("inter_buf").noconvert(),
            nb::arg("out_bufs"),
            nb::arg("output").noconvert(),
            nb::arg("combine_metadata"),
            nb::arg("num_experts"),
            nb::arg("num_cores"),
            nb::arg("grid_x"),
            nb::arg("grid_y"),
            nb::arg("reduce_recv_buf") = nb::none(),
            nb::arg("enable_fabric_reduce") = false,
            nb::arg("hidden_states_rm") = nb::none(),
            nb::arg("staging_buf") = nb::none(),
            nb::arg("enable_fabric_dispatch") = false,
            nb::arg("dispatch_metadata") = nb::none(),
            nb::arg("dispatch_target_cols") = std::vector<uint32_t>{},
            nb::arg("per_expert_dispatch_sources") = nb::none(),
            nb::arg("multi_dest_dispatch_metadata") = nb::none(),
            nb::arg("enable_fpu_combine") = false,
        });
}

}  // namespace ttnn::operations::experimental::prefill_moe_compute::detail
