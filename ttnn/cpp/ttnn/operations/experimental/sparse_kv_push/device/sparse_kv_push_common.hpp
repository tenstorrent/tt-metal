// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/distributed/types.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// One fabric channel per (link, direction), each with kSendersPerChannel sender cores (reader + writer) feeding one mux
// core that owns the router connection.
constexpr uint32_t kDirections = 4;  // E, W, N, S (the eth_chan_directions order get_neighbors returns)
constexpr uint32_t kSendersPerChannel = 2;

constexpr uint32_t cores_needed(uint32_t links) { return links * kDirections * (kSendersPerChannel + 1); }

// Rows travel in packets of pages_per_packet rows, dealt round robin over the DRAM banks. The plan lays out the
// compact slots and the push writes them, so both take the shape from here.
struct PacketShape {
    uint32_t pages_per_packet;
    uint32_t banks;
    uint32_t unit() const { return pages_per_packet * banks; }  // owner runs are padded to whole units
};
PacketShape packet_shape(tt::tt_metal::distributed::MeshDevice* mesh_device, uint32_t page_bytes);

// Sizes of a plan table for a cache of max_rows rows in blocks of stripe_chunk (word layout:
// ::sparse_push::plan_table).
struct PlanTableLayout {
    static constexpr uint32_t kBitmapOff = ::sparse_push::plan_table::kBitmapOff;
    static constexpr uint32_t kPageWords = ::sparse_push::plan_table::kPageWords;

    uint32_t bitmap_words = 0;
    uint32_t num_blocks = 0;
    uint32_t pages = 0;        // per chip
    uint32_t image_words = 0;  // the plan op's L1 image of the table: whole pages

    static PlanTableLayout make(uint32_t max_rows, uint32_t stripe_chunk) {
        PlanTableLayout l;
        l.bitmap_words = max_rows / ::sparse_push::plan_table::kRowsPerWord;
        l.num_blocks = max_rows / stripe_chunk;
        l.pages = (kBitmapOff + l.bitmap_words + kPageWords - 1) / kPageWords;
        l.image_words = l.pages * kPageWords;
        return l;
    }
};

// Shared by the plan and the push, which must be built with the same stripe_chunk and max_rows.
void validate_stripe_chunk(uint32_t stripe_chunk, uint32_t max_rows);

}  // namespace ttnn::operations::experimental::sparse_kv_push
