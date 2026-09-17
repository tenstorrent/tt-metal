// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tuple>

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
    // Broadcast only [broadcast_offset_tiles, +broadcast_num_tiles) of the sender's shard; the rest of the
    // output is left untouched. 0 num = whole shard. Cuts data moved when the caller needs a sub-range.
    uint32_t broadcast_offset_tiles = 0;
    uint32_t broadcast_num_tiles = 0;
    // L1 relay: forward each chunk straight into the downstream's L1 recv buffer (no per-hop DRAM read),
    // gated by a backward credit protocol. Default false keeps the DRAM-output relay. Experimental.
    bool use_l1_relay = false;
    // L1-relay credit window: number of recv-buffer slots (chunks in flight). 0 = auto. Deeper = more
    // pipeline overlap at the cost of L1 (slots * chunk * page_size). Ignored unless use_l1_relay.
    uint32_t num_slots = 0;

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
        bool use_l1_relay_ = false,
        uint32_t num_slots_ = 0) :
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
        use_l1_relay(use_l1_relay_),
        num_slots(num_slots_) {}

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
            use_l1_relay,
            num_slots);
    }
};

struct BroadcastRingInputs {
    Tensor input_tensor;
};

}  // namespace ttnn::prim
