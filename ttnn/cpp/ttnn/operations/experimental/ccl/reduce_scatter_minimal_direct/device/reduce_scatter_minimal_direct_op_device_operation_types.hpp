// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

// Direct (one-shot) reduce-scatter: latency-optimal sibling of reduce_scatter_minimal_unicast.
//
// Instead of a store-and-forward ring (N/2 hops, each paying a full read-add-send worker round trip),
// every device unicasts each destination's slice STRAIGHT to that destination over the fabric
// (num_hops = ring distance, no intermediate device touches the data), bumps a per-source arrival
// counter with the send's last packet, and once all N-1 contributions have landed reduces them together
// with its own slice and writes the output. Fabric traffic is ~2.3x the ring's bandwidth-optimal volume
// (a distance-h contribution crosses h links), so this is a latency play for small/medium shapes, not a
// bandwidth play. Inspired by the native all_reduce_async (mcast-then-local-reduce).
//
// The program-cache hash auto-reflects over these fields; all are stable/structural.
struct ReduceScatterMinimalDirectParams {
    int32_t dim = 0;
    MemoryConfig output_mem_config;
    std::optional<uint32_t> cluster_axis;

    // Fabric setup / per-axis geometry (an inactive axis has num_devices = 1, num_links = 0, Linear).
    tt::tt_fabric::FabricConfig fabric_config{};
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    uint32_t num_devices = 0;
    size_t packet_size = 0;
    // Resolved number of fabric links. One worker core per link, owning that link's forward AND backward
    // connection plus a contiguous sub-range of every slice's chunks. The factory clamps it to
    // chunks_per_slice so no worker is given zero work.
    uint32_t num_links = 1;

    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;
};

struct ReduceScatterMinimalDirectInputs {
    Tensor input_tensor;
    // Optional caller-provided persistent buffers (reused across invocations to skip re-allocation).
    // Index convention mirrors create_output_tensors: [0] = output slice, [1] = staging (holds the
    // incoming contributions, indexed by SOURCE device, double-buffered by invocation parity).
    std::optional<Tensor> persistent_output_tensor;
    std::optional<Tensor> persistent_staging_tensor;
};

}  // namespace ttnn::experimental::prim
