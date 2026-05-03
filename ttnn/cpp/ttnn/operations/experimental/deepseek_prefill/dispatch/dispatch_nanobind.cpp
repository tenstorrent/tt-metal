// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dispatch.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch::detail {

void bind_dispatch(nb::module_& mod) {
    ttnn::bind_function<"dispatch", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Routes input tokens to destination device dispatch buffers based on top-k expert indices.

        For each token on each source device, the kernel looks up the destination device for
        each of its top-k experts via expert_dispatch_table_tensor, then writes the token
        embedding at the token index given by expert_offsets_tensor in the destination
        device's flat dispatch buffer. Writes to the local device use NOC; writes to remote
        devices in the dispatch group use fabric. A metadata entry is written alongside each
        token for later recombination by the combine op.

        Each destination device accumulates a flat dispatch buffer: all experts_per_chip experts
        are packed contiguously in a single token dimension, with each expert's region starting
        at a TILE_HEIGHT-aligned token index.

        Args:
            input_tensor (ttnn.Tensor): Input token embeddings.
                Shape per device: (1, seq_len_per_chip, hidden_dim).
            weights_tensor (ttnn.Tensor): Router weights for each token's top-k experts.
                Shape per device: (1, seq_len_per_chip, num_experts_per_tok).
            indices_tensor (ttnn.Tensor): Top-k expert indices for each token.
                Shape per device: (1, seq_len_per_chip, num_experts_per_tok).
            expert_offsets_tensor (ttnn.Tensor): Starting token index per source device per expert
                in the destination device's flat dispatch buffer.
                Shape per device: (1, num_routed_experts).
            expert_dispatch_table_tensor (ttnn.Tensor): Maps each expert ID to the destination
                chip ID within the dispatch group. Values >= 0 are destination chip IDs; -1
                means the expert is absent from this dispatch group.
                Shape per device: (1, num_routed_experts).

        Keyword Args:
            dispatch_group_size (int): Number of devices in the dispatch group.
            experts_per_chip (int): Number of experts hosted on each destination device.
            num_routed_experts (int): Total number of routed experts across all devices.
            num_experts_per_tok (int): Number of experts each token is routed to (top-k).
            metadata_len (int): Number of fields per token in the metadata buffer (5:
                linearized_mesh_coord, token_idx, topk_idx, routed_expert, weight).
            max_dispatch_buffer_token_size (int): Total token capacity of the flat dispatch
                buffer per chip (shared across all local experts via dynamic offsets).
                Used as the in-kernel bounds check ceiling.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice ID for core allocation.
            cluster_axis (int, optional): Mesh axis along which dispatch communicates
                (0 = SP/dispatch axis). Defaults to 0.
            num_links (int, optional): Number of fabric links for remote token writes.
                Defaults to 1.
            topology (ttnn.Topology, optional): Fabric topology for remote writes.
                Defaults to Linear.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]:
                dispatched_buffer: Flat expert-centric token buffer on each destination device.
                    Shape per device: (1, 1, max_dispatch_buffer_token_size, hidden_dim).
                metadata: Per-token metadata written alongside dispatched_buffer.
                    Shape per device: (1, 1, max_dispatch_buffer_token_size, metadata_len=5).
                    Fields: [linearized_mesh_coord, token_idx, topk_idx, routed_expert, weight].
                    Used by the combine op to route processed tokens back to their origin.
        )doc",
        &dispatch,
        nb::arg("input_tensor").noconvert(),
        nb::arg("weights_tensor").noconvert(),
        nb::arg("indices_tensor").noconvert(),
        nb::arg("expert_offsets_tensor").noconvert(),
        nb::arg("expert_dispatch_table_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("dispatch_group_size"),
        nb::arg("experts_per_chip"),
        nb::arg("num_routed_experts"),
        nb::arg("num_experts_per_tok"),
        nb::arg("metadata_len"),
        nb::arg("max_dispatch_buffer_token_size"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Linear),
        nb::arg("use_l1_small_for_semaphores") = false);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_dispatch(::nanobind::module_& mod) { dispatch::detail::bind_dispatch(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
