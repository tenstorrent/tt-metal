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

        Args:
            hidden_states: [1,1,P,D] BF16 TILE_LAYOUT input activation tensor
            gate_up_weights: List of per-expert [1,1,D,D_FF] BFP4_b weight tensors
            down_weights: List of per-expert [1,1,D_FF/2,D] BFP4_b weight tensors
            pkt_buf: [1,1,P,D] BF16 scratch buffer for dispatch
            inter_buf: [1,1,P,D_FF/2] BF16 scratch buffer for intermediate
            out_bufs: List of per-expert [1,1,P,D] BF16 output scratch buffers
            output: [1,1,P,D] BF16 pre-allocated output tensor (zero-filled)
            per_device_combine_metadata: Per-device packed routing args for combine kernel
            num_experts: Number of experts (1-4)
            num_cores: Number of matmul cores
            grid_x: Matmul core grid X dimension
            grid_y: Matmul core grid Y dimension
            hidden_states_rm: Optional [1,1,P,D] BF16 ROW_MAJOR for fabric dispatch
            staging_buf: Optional [1,1,P,D] BF16 ROW_MAJOR fabric receive buffer
            enable_fabric_dispatch: Enable fabric token dispatch (default False)
            dispatch_metadata: Per-device dispatch routing metadata
            enable_fabric_return: Enable fabric return (default False)
            return_metadata: Per-device return routing metadata
            recv_staging_buf: Optional ROW_MAJOR staging buffer for fabric return receives
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
            nb::arg("per_device_combine_metadata"),
            nb::arg("num_experts"),
            nb::arg("num_cores"),
            nb::arg("grid_x"),
            nb::arg("grid_y"),
            nb::arg("hidden_states_rm") = nb::none(),
            nb::arg("staging_buf") = nb::none(),
            nb::arg("enable_fabric_dispatch") = false,
            nb::arg("dispatch_metadata") = nb::none(),
            nb::arg("enable_fabric_return") = false,
            nb::arg("return_metadata") = nb::none(),
            nb::arg("recv_staging_buf") = nb::none(),
            nb::arg("return_metadata_tensor") = nb::none(),
        });
}

}  // namespace ttnn::operations::experimental::prefill_moe_compute::detail
