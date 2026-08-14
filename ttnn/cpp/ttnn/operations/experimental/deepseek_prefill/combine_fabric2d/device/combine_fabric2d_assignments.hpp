// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "combine_fabric2d_placement.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// One unit of work for one stream, in execution order.
//
// A relay pushes an incoming forwarding chunk one hop further, whole. Otherwise the work is this chip's own
// tokens for one destination chip, narrowed to a fraction of each run: halved between the routing planes for
// destinations nearer than the opposite chip, quartered across all streams for the opposite chip, which is
// equally far in both directions.
//
// Where a run starts and how long it is are NOT here. Run boundaries live in the caller's control tensors
// and are only knowable on device, so an assignment names the run (`dst_row`) and the share
// (`split_idx`/`split_count`) and the kernel resolves both to page ranges.
struct Assignment {
    bool is_relay = false;
    uint32_t relay_chunk = 0;  // is_relay: which chunk of this stream's quarter
    uint32_t dst_chip_id = 0;  // !is_relay: fabric name of the destination chip
    uint32_t dst_row = 0;      // !is_relay: that chip's row on the ring, indexing expert_offsets
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Work for every stream on `coord`. `ring_chip_ids` holds the fabric chip id of each row of the ring, so
// this needs nothing from the mesh API.
std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_row, uint32_t num_links);

// Relay chunks a stream receives, which is also how many its upstream neighbour emits into this stream's
// quarter. Equals (own forwarding) + (re-forwarded), so upstream writer and downstream reader agree on a
// quarter's chunk count without exchanging anything.
constexpr uint32_t relay_chunks_per_stream(uint32_t ring_extent) {
    const uint32_t m = ring_extent / 2;
    return m * (m - 1) / 2;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
