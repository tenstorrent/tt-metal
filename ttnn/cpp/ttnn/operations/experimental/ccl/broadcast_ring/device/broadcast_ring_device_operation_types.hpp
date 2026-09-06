// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tuple>
#include <vector>

namespace ttnn::prim {

// One-sender ring broadcast: the shard at `sender_ring_index` along `cluster_axis` is delivered to every
// device on that ring line (manual per-hop unicast relay, FABRIC_1D / FABRIC_1D_RING). Runs independently
// per line on the orthogonal axis, so a tp-sharded tensor broadcasts each tp row's own data. v1: single
// sender, one-way around the ring (needs the wrap link -> Ring topology). See models/tt_dit/docs/
// broadcast_ring_design.md.
struct BroadcastRingParams {
    uint32_t sender_ring_index = 0;  // index along cluster_axis whose shard is broadcast
    uint32_t cluster_axis = 0;       // ring axis to broadcast along
    uint32_t num_links = 0;
    uint32_t ring_size = 0;  // devices along cluster_axis
    MemoryConfig output_mem_config;
    tt::tt_fabric::Topology topology{};
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    uint32_t chunk_size_tiles = 0;  // tiles per relay chunk; 0 = auto. Tuning knob.
    // Broadcast range: broadcast_num_blocks blocks of broadcast_num_tiles pages, the b-th block starting
    // at flat page broadcast_offset_tiles + b*broadcast_stride_pages; the rest of the output is left
    // untouched. num_blocks==1 (default) is the plain contiguous range [offset, +num) (0 num = whole
    // shard). Blocks let a caller express a dim-2 (seq) sub-range of a [B, H, S, E] tiled tensor -- one
    // block per (b, h) with stride = S_rows*E_cols pages -- which a flat contiguous range cannot once
    // E/32 > 1 or B*H > 1. Blocked ranges are L1-relay only.
    uint32_t broadcast_offset_tiles = 0;
    uint32_t broadcast_num_tiles = 0;
    uint32_t broadcast_stride_pages = 0;
    uint32_t broadcast_num_blocks = 1;
    // L1 relay: forward each chunk straight into the downstream's L1 recv buffer (no per-hop DRAM read),
    // gated by a backward credit protocol. Default false keeps the DRAM-output relay. Experimental.
    bool use_l1_relay = false;
    // L1-relay credit window: number of recv-buffer slots (chunks in flight). 0 = auto. Deeper = more
    // pipeline overlap at the cost of L1 (slots * chunk * page_size). Ignored unless use_l1_relay.
    uint32_t num_slots = 0;
    // Caller-owned global semaphores {recv, cred_fwd, cred_bwd} (rotated ping-pong by the caller). The op
    // uses these instead of creating its own, so under tracing -- where the program factory runs once at
    // capture -- each call still gets its own semaphore rather than a single baked one shared by every
    // replay. Empty falls back to the op creating them internally (untraced use). Excluded from the cache
    // key (below): the semaphore is a runtime arg refreshed by override_runtime_arguments.
    std::vector<tt::tt_metal::GlobalSemaphore> multi_device_global_semaphore;

    BroadcastRingParams(
        uint32_t sender_ring_index_,
        uint32_t cluster_axis_,
        uint32_t num_links_,
        uint32_t ring_size_,
        const MemoryConfig& output_mem_config_,
        tt::tt_fabric::Topology topology_,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id_,
        uint32_t chunk_size_tiles_ = 0,
        uint32_t broadcast_offset_tiles_ = 0,
        uint32_t broadcast_num_tiles_ = 0,
        uint32_t broadcast_stride_pages_ = 0,
        uint32_t broadcast_num_blocks_ = 1,
        bool use_l1_relay_ = false,
        uint32_t num_slots_ = 0,
        std::vector<tt::tt_metal::GlobalSemaphore> multi_device_global_semaphore_ = {}) :
        sender_ring_index(sender_ring_index_),
        cluster_axis(cluster_axis_),
        num_links(num_links_),
        ring_size(ring_size_),
        output_mem_config(output_mem_config_),
        topology(topology_),
        sub_device_id(sub_device_id_),
        chunk_size_tiles(chunk_size_tiles_),
        broadcast_offset_tiles(broadcast_offset_tiles_),
        broadcast_num_tiles(broadcast_num_tiles_),
        broadcast_stride_pages(broadcast_stride_pages_),
        broadcast_num_blocks(broadcast_num_blocks_),
        use_l1_relay(use_l1_relay_),
        num_slots(num_slots_),
        multi_device_global_semaphore(std::move(multi_device_global_semaphore_)) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "sender_ring_index",
        "cluster_axis",
        "num_links",
        "ring_size",
        "output_mem_config",
        "topology",
        "sub_device_id",
        "chunk_size_tiles",
        "broadcast_offset_tiles",
        "broadcast_num_tiles",
        "broadcast_stride_pages",
        "broadcast_num_blocks",
        "use_l1_relay",
        "num_slots");
    auto attribute_values() const {
        return std::make_tuple(
            sender_ring_index,
            cluster_axis,
            num_links,
            ring_size,
            output_mem_config,
            topology,
            sub_device_id,
            chunk_size_tiles,
            broadcast_offset_tiles,
            broadcast_num_tiles,
            broadcast_stride_pages,
            broadcast_num_blocks,
            use_l1_relay,
            num_slots);
    }
};

struct BroadcastRingInputs {
    Tensor input_tensor;
    // Optional caller-owned output buffer. Required for trace-safety: a captured trace bakes the output
    // address, so a fresh per-call output (create_device_tensor) is freed after capture and clobbered on
    // replay. Pass a persistent buffer (same spec as the input) to reuse a stable address across replays.
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::prim
