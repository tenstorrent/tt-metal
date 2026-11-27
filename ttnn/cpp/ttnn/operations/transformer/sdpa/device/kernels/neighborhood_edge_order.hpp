// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "neighborhood_chunk_layout.hpp"

namespace ttnn::transformer::neighborhood::chunk_layout {

// The order a core visits its work items in (DIFFVAE_NA_EDGE_ORDER). The reader and the writer
// must walk the same order, so both use this.
//
// A chunk's mask block depends on where its windows clamp at the volume edge, and on an edge shard
// the first and last `depth` chunks of every H column and W row clamp differently from the rest.
// In plain index order (W fastest) a core crosses those edge chunks on every row, so the persistent
// mask block is rewritten several times per row. This order visits the core's range grouped by
// (H edge position, W edge position): all interior chunks first, then each edge position's chunks,
// each group in index order. Chunks of one group share their H and W clamp, so the block is
// rewritten about once per group instead. Correctness never depends on the grouping: the reader
// still compares every chunk's WindowClamp before reusing the block.
//
// depth_height == depth_width == 0 keeps plain index order.
template <uint32_t depth_height, uint32_t depth_width, uint32_t chunk_count>
class EdgeGroupedOrder {
public:
    EdgeGroupedOrder(uint32_t start, uint32_t count, ShapeInChunks volume_chunks) :
        start_(start), end_(start + count), cursor_(start), volume_chunks_(volume_chunks) {}

    // The next work item. Call exactly `count` times.
    uint32_t next() {
        if constexpr (depth_height == 0 && depth_width == 0) {
            return cursor_++;
        } else {
            while (true) {
                while (cursor_ < end_) {
                    const uint32_t item = cursor_++;
                    const uint32_t key = group_of(item);
                    if (key == group_) {
                        return item;
                    }
                    if (key > group_ && key < next_group_) {
                        next_group_ = key;
                    }
                }
                group_ = next_group_;
                next_group_ = 0xFFFFFFFFu;
                cursor_ = start_;
            }
        }
    }

private:
    // 0 for an interior position, 1..depth for the low edge, depth+1..2*depth for the high edge.
    template <uint32_t depth>
    static uint32_t edge_position(uint32_t position, uint32_t extent) {
        if (position < depth) {
            return position + 1;
        }
        if (position + depth >= extent) {
            return depth + extent - position;
        }
        return 0;
    }

    // Items of different heads or batches never share a group.
    uint32_t group_of(uint32_t item) const {
        const auto chunk = linear_to_point3<Unit::Chunks>(item % chunk_count, volume_chunks_);
        return ((item / chunk_count) << 16) |
               (edge_position<depth_height>(chunk.height(), volume_chunks_.height()) << 8) |
               edge_position<depth_width>(chunk.width(), volume_chunks_.width());
    }

    uint32_t start_;
    uint32_t end_;
    uint32_t cursor_;
    uint32_t group_ = 0;
    uint32_t next_group_ = 0xFFFFFFFFu;
    ShapeInChunks volume_chunks_;
};

}  // namespace ttnn::transformer::neighborhood::chunk_layout
