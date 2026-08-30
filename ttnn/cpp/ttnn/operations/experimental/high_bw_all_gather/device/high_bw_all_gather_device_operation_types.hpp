// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/snake_ring.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

// The device-operation framework reflects over this aggregate for the default
// program-cache key. Keep only stable structural values here; semaphores and
// raw pointers belong in the program factory's shared variables.
struct HighBwAllGatherParams {
    int32_t dim = 0;
    MemoryConfig output_mem_config;
    uint32_t cluster_axis = 0;
    // With no public cluster_axis, linearize the complete 2D mesh into a
    // direct-neighbor snake ring. cluster_axis is ignored in this mode.
    bool linearized_mesh_ring = false;
    ttnn::ccl::snake_ring::Orientation snake_ring_orientation = ttnn::ccl::snake_ring::Orientation::Row;

    // Fabric setup info
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;
    // Per-axis info (an inactive axis has num_devices = 1, num_links = 0, and Linear topology)
    std::array<tt::tt_fabric::Topology, 2> axis_topology{};
    std::array<uint32_t, 2> axis_num_devices{};
    std::array<uint32_t, 2> axis_num_links{};
    uint32_t num_devices = 0;  // number of devices participating in the collective
    uint32_t num_links = 0;
    uint32_t mesh_rows = 0;
    uint32_t mesh_cols = 0;
    size_t packet_size = 0;
    // Host-proved structural eligibility for the native store-and-forward
    // transport. Under Fabric2D every logical edge, including ring wrap, must
    // be one direct physical neighbor hop.
    bool neighbor_unicast_eligible = false;
    // Hash of the complete directed physical neighbor plan. Fabric routing
    // arguments are baked into cached programs, so eligibility alone is not a
    // sufficient cache discriminator when the physical plan changes.
    std::optional<uint64_t> neighbor_route_plan_hash;

    // Worker-core selection.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    std::optional<CoreRangeSet> sub_core_grid;

    // Optional runtime controls for gathering one slot of a persistent cache into a maximum-capacity
    // output buffer. `gathered_dim_size` is the global (post-gather) valid extent along dim; each rank's
    // local prefix occupies that rank's fixed worst-case output slot, and the allocation remains full size.
    // Their values are hash-excluded and patch only indexed kernel runtime arguments, so slots/prefixes
    // reuse one cached program.
    std::optional<uint32_t> input_batch_index;
    std::optional<uint32_t> gathered_dim_size;

    // Shared paged-cache geometry. The table itself is a tensor argument so its address is repatched.
    uint32_t kv_cache_page_size = 32;
    uint32_t kv_cache_num_layers = 1;
    uint32_t kv_cache_layer_idx = 0;
};

struct HighBwAllGatherInputs {
    Tensor input_tensor;
    Tensor output_tensor;
    std::optional<Tensor> page_bundle_indices;

    bool has_paged_input() const { return page_bundle_indices.has_value(); }
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather
