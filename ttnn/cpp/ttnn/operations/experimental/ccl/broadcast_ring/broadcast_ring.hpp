// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <vector>

namespace ttnn {

// Broadcast the shard at `sender_ring_index` (along `cluster_axis`) to every device on that ring line,
// via manual per-hop unicast relay over FABRIC_1D / FABRIC_1D_RING. The op runs independently on each
// line parallel to the orthogonal axis, so a tp-sharded input broadcasts each orthogonal row's own data
// (the per-line semantics ttnn.broadcast lacks). Output has the same shape/spec as the input; every
// device on the line ends holding the sender shard's data.
//
// v1 constraints: single sender; one-way around the ring (requires the wrap link -> Ring topology).
ttnn::Tensor broadcast_ring(
    const ttnn::Tensor& input_tensor,
    uint32_t sender_ring_index,
    uint32_t cluster_axis,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    uint32_t chunk_size_tiles = 0,        // tiles per relay chunk; 0 = auto. Tuning knob.
    uint32_t broadcast_offset_tiles = 0,  // broadcast only [offset, +num) of the sender's shard; 0 num = whole.
    uint32_t broadcast_num_tiles = 0,
    // Blocked range (L1 relay only): num_blocks blocks of broadcast_num_tiles pages, block b starting at
    // offset + b*stride. Expresses a dim-2 (seq) sub-range of a [B, H, S, E] tiled tensor (one block per
    // (b, h), stride = S_rows*E_cols), which a flat contiguous range cannot. num_blocks<=1 = contiguous.
    uint32_t broadcast_stride_pages = 0,
    uint32_t broadcast_num_blocks = 1,
    bool use_l1_relay = false,  // relay through L1 (no per-hop DRAM read) via a credit protocol. Experimental.
    uint32_t num_slots = 0,     // L1-relay credit window (recv-buffer slots); 0 = auto. Tuning knob.
    // Caller-owned output buffer (same shape as input). Required under tracing: a fresh per-call output is
    // clobbered on trace replay, so pass a persistent buffer for a stable baked address.
    const std::optional<ttnn::Tensor>& persistent_output_buffer = std::nullopt,
    // Caller-owned ping-pong global semaphores {recv, cred_fwd, cred_bwd}. Required under tracing so each
    // call gets its own semaphore instead of one baked into the single captured program (see the op).
    const std::vector<tt::tt_metal::GlobalSemaphore>& multi_device_global_semaphore = {});

}  // namespace ttnn
