// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_common.hpp"

#include <algorithm>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operation.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

void validate_stripe_chunk(uint32_t stripe_chunk, uint32_t max_rows) {
    TT_FATAL(
        stripe_chunk > 0 && stripe_chunk % 32 == 0 && stripe_chunk <= ::sparse_push::kMaxStripeChunk,
        "sparse KV push stripe_chunk {} must be a positive multiple of 32, at most {}",
        stripe_chunk,
        ::sparse_push::kMaxStripeChunk);
    TT_FATAL(
        max_rows > 0 && max_rows % stripe_chunk == 0,
        "sparse KV push max_rows {} must be a positive multiple of stripe_chunk {}",
        max_rows,
        stripe_chunk);
}

PacketShape packet_shape(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t page_bytes) {
    const auto max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    TT_FATAL(
        page_bytes > 0 && page_bytes <= max_payload,
        "the sparse KV push needs a row page of 1 to {} bytes (the fabric's max payload), got {}",
        max_payload,
        page_bytes);
    return PacketShape{
        .pages_per_packet = max_payload / page_bytes, .banks = static_cast<uint32_t>(mesh_device->num_dram_channels())};
}

uint32_t plan_cores(tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const auto grid = mesh_device->compute_with_storage_grid_size();
    return grid.x * grid.y;
}

bool plan_fits(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t max_rows) {
    return max_rows % ::sparse_push_plan::kChunkRows == 0 &&
           tt::div_up(max_rows / ::sparse_push_plan::kChunkRows, plan_cores(mesh_device)) <=
               ::sparse_push_plan::kMaxChunksPerCore;
}

tt::tt_metal::TensorTopology rows_over_every_chip(const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    using tt::tt_metal::distributed::MeshCoordinate;
    using tt::tt_metal::distributed::MeshMapperConfig;
    const tt::tt_metal::distributed::MeshCoordinateRange chips(mesh_shape);
    ttsl::SmallVector<MeshMapperConfig::Placement> placements(mesh_shape.dims(), MeshMapperConfig::Shard{2});
    return tt::tt_metal::TensorTopology(
        mesh_shape, std::move(placements), std::vector<MeshCoordinate>(chips.begin(), chips.end()));
}

uint32_t output_rows(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t distinct_rows,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t page_bytes) {
    validate_stripe_chunk(stripe_chunk, max_rows);
    const uint32_t num_devices = mesh_device->shape().mesh_size();
    const uint32_t unit = packet_shape(mesh_device, page_bytes).unit();  // each owner's run is padded to whole units
    const uint32_t padded_distinct = std::min(distinct_rows, max_rows) + num_devices * unit;
    const uint32_t owner_rows = tt::div_up(max_rows / stripe_chunk, num_devices) * stripe_chunk;
    const uint32_t all_full = num_devices * tt::round_up(owner_rows, unit);
    return std::min(padded_distinct, all_full);
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
