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

    // TRACE-SAFE slot select. `input_batch_index` above is a host runtime argument: the program-cache
    // path re-patches the reader's input page base on every dispatch, but a ttnn trace REPLAY never runs
    // that host patch, so every replay would re-read the slot that happened to be live at capture time.
    // For chunked prefill that is silent corruption, not an error -- the KV write is metadata-driven and
    // lands in the right slot while the gather reads the captured one.
    //
    // On the metadata path the caller hands over a 1-element uint32 tensor holding the USER id, and the
    // reader recomposes the flat cache slot on-device as
    //     batch_index = user_id * batch_slot_num_layers + batch_slot_layer_idx
    // mirroring ttMLA._cache_batch_idx (the cache batch dim is user-major). The two layer terms are
    // RUNTIME arguments and deliberately NOT hashed: this op is shape-identical across layers, so all
    // layers share ONE cached program, and hashing the layer index would fork it per layer -- each fork
    // allocating two more global semaphores, which exhausts a tight L1_SMALL region. Freezing them into a
    // capture is still correct: they are layer-constant, and each layer is its own captured op instance.
    uint32_t batch_slot_num_layers = 1;
    uint32_t batch_slot_layer_idx = 0;

    // TRACE-SAFE active extent. `gathered_dim_size` above is likewise a host runtime argument, and it
    // GROWS every chunk, so a replay would re-gather only the captured chunk's prefix and leave the rest
    // of the KV unread. On the metadata path the caller instead hands over the chunk's start position and
    // the reader derives the extent on-device:
    //     populated = start + gathered_slab_global
    //     gathered   = min(round_up(populated, gathered_slab_global), full_extent)
    // `gathered_slab_global` is the block-cyclic slab width in gathered-dim elements (chunk_local * sp).
    // It is structural -- identical for every chunk and every layer -- so it is hashed.
    uint32_t gathered_slab_global = 0;
};

struct HighBwAllGatherInputs {
    Tensor input_tensor;
    Tensor output_tensor;
    // Fused slot select (see HighBwAllGatherParams::batch_slot_*). Mutually exclusive with the scalar
    // `input_batch_index`. 1-element uint32 ROW_MAJOR DRAM tensor holding the user/slot id.
    std::optional<Tensor> input_batch_index_tensor{std::nullopt};
    bool has_batch_index_metadata() const { return input_batch_index_tensor.has_value(); }
    // Fused active-extent select (see HighBwAllGatherParams::gathered_slab_global). Mutually exclusive
    // with the scalar `gathered_dim_size`. 1-element uint32 ROW_MAJOR DRAM tensor holding this chunk's
    // start position in the gathered dim.
    std::optional<Tensor> gathered_prefix_tensor{std::nullopt};
    bool has_gathered_prefix_metadata() const { return gathered_prefix_tensor.has_value(); }
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather
