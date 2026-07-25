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
            input_tensor (ttnn.Tensor): Input token embeddings. Dtype may be BFLOAT16 or
                FP8_E4M3 (FP8_E4M3 is Blackhole-only). TILE layout converts to the output
                dtype via the compute packer; ROW_MAJOR layout is a byte copy and requires
                the input dtype to match the output dtype.
                Shape per device: (1, seq_len_per_chip, hidden_dim).
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
            metadata_len (int): Number of fields per token in the metadata buffer. The first 3 are
                routing fields (linearized_mesh_coord, token_idx, topk_idx).
                When scales_tensor is provided, metadata_len must be 3 + (emb_dim/numbers_per_scale_block): each token's
                fp32 per-128-block (numbers_per_scale_block) scales are appended as fields 3.. (stored as int32 bit-patterns).
            max_dispatch_buffer_token_size (int): Total token capacity of the flat dispatch
                buffer per chip (shared across all local experts via dynamic offsets).
                Used as the in-kernel bounds check ceiling.
            padding_config (ttnn.Tensor, optional): Per-device [local_real_tokens, pad_side]
                config (uint32, ROW_MAJOR, shape (1, 2) per device). When provided, the
                dispatch kernels read it on device and bound their token loop to the real
                (unpadded) tokens instead of the full seq_len_per_chip. Must be the same
                tensor the gate used to sentinel-mark padded tokens. Defaults to None
                (process the full token range).
            scales_tensor (ttnn.Tensor, optional): Per-token fp8 scales (ROW_MAJOR,
                shape (1, seq_len_per_chip, emb_dim/numbers_per_scale_block) per device) produced by
                per_token_cast_to_fp8 alongside the fp8 input. When provided (fp8 ROW_MAJOR
                input only), each token's scales are copied into the metadata tail so the
                routed buffer can be dequantized downstream. Requires metadata_len ==
                3 + emb_dim/numbers_per_scale_block. Defaults to None. Blackhole-only.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice ID for core allocation.
            cluster_axis (int, optional): Mesh axis along which dispatch communicates
                (0 = SP/dispatch axis). Defaults to 0.
            num_links (int, optional): Number of fabric links for remote token writes.
                Defaults to 1.
            topology (ttnn.Topology, optional): Fabric topology for remote writes.
                Defaults to Linear.
            use_l1_small_for_semaphores (bool, optional): Allocate the workload's
                GlobalSemaphores in L1_SMALL instead of L1. Defaults to False.
            fp8_output (bool, optional): Pack the dispatched buffer as Fp8_e4m3
                (DataType::FP8_E4M3). Blackhole-only (not supported on Wormhole_B0). With
                TILE input the packer converts any input dtype; with ROW_MAJOR input (a byte
                copy) fp8 output requires fp8 input. Defaults to False.
            fp8_scaled_input (bool, optional): Enable the fp8-scaled-input path. When True, the
                input must be fp8 (FP8_E4M3) ROW_MAJOR and scales_tensor must be provided; each
                token's fp32 per-128-block (numbers_per_scale_block) scales are copied into the metadata tail (fields 3..),
                requiring metadata_len == 3 + emb_dim/numbers_per_scale_block. Blackhole-only. Defaults to False.
            num_workers_per_sender (int, optional): Number of worker cores per
                sender. Applies to both TILE and ROW_MAJOR input (both layouts run on
                the same worker architecture; ROW_MAJOR reads rows without untilizing).

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]:
                dispatched_buffer: Flat expert-centric token buffer on each destination device.
                    Shape per device: (1, 1, max_dispatch_buffer_token_size, hidden_dim).
                metadata: Per-token metadata written alongside dispatched_buffer.
                    Shape per device: (1, 1, max_dispatch_buffer_token_size, metadata_len).
                    metadata_len is 3 by default — fields [linearized_mesh_coord, token_idx,
                    topk_idx] — and grows to 3 + emb_dim/numbers_per_scale_block when
                    scales_tensor is dispatched, with fields 3.. holding the per-128-block (numbers_per_scale_block) fp32
                    scale tail (int32 bit-pattern).
                    Used by the combine op to route processed tokens back to their origin.
        )doc",
        &dispatch,
        nb::arg("input_tensor").noconvert(),
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
        nb::arg("padding_config") = nb::none(),
        nb::arg("scales_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Linear),
        nb::arg("use_l1_small_for_semaphores") = false,
        nb::arg("fp8_output") = false,
        nb::arg("fp8_scaled_input") = false,
        nb::arg("num_workers_per_sender") = 2);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_dispatch(::nanobind::module_& mod) { dispatch::detail::bind_dispatch(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
