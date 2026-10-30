// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_common.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
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

}  // namespace ttnn::operations::experimental::sparse_kv_push
