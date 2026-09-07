// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "dispatch_fabric2d_placement.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// One unit of work for one stream, in execution order: this chip's own tokens for ONE destination chip,
// narrowed to a fraction of what it owes that chip. Halved between the routing planes for destinations
// nearer than the diametrically opposite chip, quartered across all streams for that chip, which is
// equally far in both directions.
//
// Which tokens those are is NOT here. A token's destination is data-dependent (indices -> dispatch
// table), so the reader builds the per-destination token list on device and this only names the
// destination (`dst_chip_id`) and the share (`split_idx`/`split_count`).
struct Assignment {
    uint32_t dst_chip_id = 0;  // fabric chip id of the destination
    uint32_t dst_row = 0;      // that chip's position on the dispatch axis
    uint32_t split_idx = 0;
    uint32_t split_count = 1;
};

// Work for every stream on one chip. `ring_chip_ids` holds the fabric chip id of each position on the
// dispatch axis, so this needs nothing from the mesh API.
std::map<StreamId, std::vector<Assignment>> generate_assignments(
    const std::vector<uint32_t>& ring_chip_ids, uint32_t my_row, uint32_t num_links);

// Own assignments a stream carries: one per destination it is nearer to in its own direction, plus its
// share of the diametrically opposite chip.
constexpr uint32_t own_assignments_per_stream(uint32_t ring_extent) { return ring_extent / 2; }

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
