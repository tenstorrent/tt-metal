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

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail {

void bind_unified_routed_expert_ffn(nb::module_& mod) {
    // Activation variant for the fused routed-expert FFN. Registered before the
    // function bindings so it can serve as a default kwarg value.
    nb::enum_<RoutedExpertActivation>(mod, "RoutedExpertActivation")
        .value("Silu", RoutedExpertActivation::Silu)
        .value("SwiGluOai", RoutedExpertActivation::SwiGluOai)
        .value("SituGlu", RoutedExpertActivation::SituGlu)
        .value("ClampedSiluGlu", RoutedExpertActivation::ClampedSiluGlu);

    // The worker rectangle this op fixes regardless of device grid. Exported so a hybrid forward can
    // pass moe_fused_swiglu the same grid -- that op defaults to the whole device instead -- without
    // a second copy of the numbers on the Python side.
    mod.attr("UNIFIED_ROUTED_EXPERT_CORE_GRID") = CoreCoord{kCoreGridX, kCoreGridY};

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
                Silu (default, DeepSeek), SwiGluOai (clamped, MiniMax-M3 / gpt-oss),
                SituGlu (tanh-capped, Kimi K3; Blackhole only), or ClampedSiluGlu
                (silu(min(gate,10)) * clamp(up,+/-10), DeepSeek V4; Blackhole only).

            l1_end (int, optional): L1 address the op's circular buffers must end below; 0 (default)
                uses the whole L1. Pass the lowest address of buffers kept resident in L1
                (MeshDevice.lowest_occupied_compute_l1_address()) to fit the blocking around them.
            x_row_index (ttnn.Tensor, optional): UINT32 ROW_MAJOR row of one entry per buffer row. With it,
                dispatched_buffer is the token-ordered ROW_MAJOR input and buffer row r reads its row
                x_row_index[r] (no dispatch gather); the output has one row per entry.
            output_row_major (bool, optional): ROW_MAJOR bf16 output instead of TILE bf8 (ROW_MAJOR
                dispatched_buffer only). Rows outside every expert region stay uninitialized.
            zero_output_row (int, optional): with output_row_major, one output row the op sets to zeros.
            weights_tile_transposed (bool, optional): every weight tile is stored transposed within itself (the
                tile grid is unchanged); the matmuls transpose it back as they unpack it. A block-float weight then
                shares one exponent per 16 consecutive input rows of one output column instead of per 16 columns.
            weight_pages (ttnn.Tensor, optional): table-addressed weights. UINT32 ROW_MAJOR DRAM tensor of shape
                [experts, 2 * (2 * weights_n_gate_up + weights_n_down)], one row (page) per local expert: the gate
                columns' entries, then the up columns', then the down columns', each entry two words
                (dram_bank_id, byte_offset) of the column's tile k = 0. Tile k is at byte_offset + k * k_stride in the
                same bank; byte_offset is relative to the source tensor's buffer address. gate_projs / up_projs /
                down_projs must then be empty; the number of local experts is the table's row count.
            gate_up_weight_source (ttnn.Tensor, optional): DRAM tensor whose buffer holds the gate and up tiles (with
                weight_pages; only its address is used).
            down_weight_source (ttnn.Tensor, optional): DRAM tensor whose buffer holds the down tiles.
            gate_up_k_stride (int, optional): bytes between tile k and k + 1 of one gate / up column.
            down_k_stride (int, optional): bytes between tile k and k + 1 of one down column.
            weights_n_gate_up (int, optional): gate (= up) tile columns per expert.
            weights_n_down (int, optional): down tile columns per expert (= emb / 32).
            weights_k_gate_up (int, optional): gate / up K tiles (= emb / 32).
            weights_k_down (int, optional): down K tiles (= weights_n_gate_up).
            weights_dtype (ttnn.DataType, optional): the weight tiles' format (bfloat4_b, bfloat8_b or bfloat16).

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
        nb::arg("l1_end") = 0,
        nb::arg("x_row_index") = nb::none(),
        nb::arg("output_row_major") = false,
        nb::arg("zero_output_row") = std::numeric_limits<uint32_t>::max(),
        nb::arg("weights_tile_transposed") = false,
        nb::arg("weight_pages") = nb::none(),
        nb::arg("gate_up_weight_source") = nb::none(),
        nb::arg("down_weight_source") = nb::none(),
        nb::arg("gate_up_k_stride") = 0,
        nb::arg("down_k_stride") = 0,
        nb::arg("weights_n_gate_up") = 0,
        nb::arg("weights_n_down") = 0,
        nb::arg("weights_k_gate_up") = 0,
        nb::arg("weights_k_down") = 0,
        nb::arg("weights_dtype") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_unified_routed_expert_ffn(::nanobind::module_& mod) {
    unified_routed_expert_ffn::detail::bind_unified_routed_expert_ffn(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
