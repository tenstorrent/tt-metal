// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "combine.hpp"
#include "device/combine_connectivity.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine::detail {

void bind_combine(nb::module_& mod) {
    ttnn::bind_function<"combine", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Routes expert-processed tokens back to their origin devices and accumulates weighted contributions at each token's original position.

        For each entry in dispatched_buffer, the kernel reads the corresponding metadata entry
        to determine the origin device, original token index, top-k slot, and router weight.
        It then writes the weighted expert output back to the origin device's output buffer:
        locally via NOC if the origin is the same device, or remotely via fabric if the origin
        is a different device in the dispatch group. This is the inverse of the dispatch op.

        Each device accumulates a token-centric output buffer: for each token slot, up to
        num_experts_per_tok expert contributions are written at the corresponding top-k index.
        Only the token slots corresponding to experts in this dispatch group are populated;
        slots for experts from other dispatch groups contain uninitialized values.

        Args:
            dispatched_buffer (ttnn.Tensor): Expert-processed token embeddings produced by TtRoutedExpert.
                Shape per device: (1, 1, max_dispatch_buffer_token_size, hidden_dim).
                BFLOAT16 ROW_MAJOR.
            dispatched_metadata (ttnn.Tensor): Per-token routing metadata produced by the dispatch op.
                Shape per device: (1, 1, max_dispatch_buffer_token_size, metadata_len=5).
                INT32 ROW_MAJOR. Fields per token: [linearized_mesh_coord, token_idx, topk_idx, routed_expert, weight].
            expert_token_counts (ttnn.Tensor): Number of tokens dispatched to each expert, used to bound
                the valid range of token slots read per expert in dispatched_buffer.
                Shape per device: (1, 1, num_routed_experts). INT32 or UINT32 ROW_MAJOR.
            expert_region_offsets (ttnn.Tensor): Expert region offsets (shared across source
                devices in a dispatch group) giving the start position of each expert's region
                in the dispatched_buffer. Same shape/layout as expert_token_counts. Produced by
                ttnn.experimental.deepseek_prefill.offset_cumsum.
                Shape per device: (1, 1, num_routed_experts). INT32 or UINT32 ROW_MAJOR.

        Keyword Args:
            dispatch_group_size (int): Number of devices in the dispatch group.
            experts_per_chip (int): Number of experts hosted on each device.
            num_experts_per_tok (int): Number of experts each token is routed to (top-k).
            seq_len_per_chip (int): Number of tokens on each device (output token dimension size).
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Must be interleaved
                (L1 or DRAM). Defaults to the memory config of dispatched_buffer.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice ID for core allocation. Defaults to None.
            cluster_axis (int, optional): Mesh axis along which combine communicates
                (0 = SP/dispatch axis). Defaults to 0.
            num_links (int, optional): Number of fabric links for remote token writes.
                Defaults to 1.
            topology (ttnn.Topology, optional): Fabric topology for remote writes (Linear or Ring).
                Defaults to Linear.
            init_zeros (bool, optional): Whether to zero-initialize the output buffer before writing.
                Defaults to True.
            use_fp8_combine (bool, optional): When True, emit the combined output in fp8_e4m3.
                Requires Blackhole hardware. Defaults to False.

        Returns:
            ttnn.Tensor:
                Combined token embeddings with weighted expert contributions accumulated at each
                token's original top-k slot.
                Shape per device: (1, 1, seq_len_per_chip, num_experts_per_tok, hidden_dim).
                BFLOAT16 ROW_MAJOR. Token slots corresponding to experts outside this dispatch
                group contain uninitialized values.
        )doc",
        &combine,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("dispatched_metadata").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::kw_only(),
        nb::arg("dispatch_group_size"),
        nb::arg("experts_per_chip"),
        nb::arg("num_experts_per_tok"),
        nb::arg("seq_len_per_chip"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Linear),
        nb::arg("init_zeros") = true,
        nb::arg("use_l1_small_for_semaphores") = false,
        nb::arg("use_fp8_combine") = false);

    // [debug] Dump the per-device combine connectivity (eth-to-eth links + sender/untilizer placement) that the
    // program factory captured when it last built the op. Call AFTER the combine op has run. Writes one
    // device_{id}_connectivity.txt per device into out_dir (default "generated/combine_flow_log"), alongside the
    // [rxlog]/[txlog] flow-control traces so the topology and the traces sit together.
    ttnn::bind_function<"dump_combine_connectivity", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Dump the combine op's per-eth-to-eth connectivity and sender/untilizer core placement (physical NOC0
            + NOC1 coords, routing plane, and destination device per eth link) to one text file per device.
            Call after the combine op has run. out_dir defaults to "generated/combine_flow_log".
        )doc",
        +[](tt::tt_metal::distributed::MeshDevice* mesh_device, const std::string& out_dir) {
            ttnn::operations::experimental::deepseek_prefill::combine::dump_combine_connectivity(*mesh_device, out_dir);
        },
        nb::arg("mesh_device"),
        nb::arg("out_dir") = std::string(""));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_combine(::nanobind::module_& mod) { combine::detail::bind_combine(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
