// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Segment-mapped TILE reshape for any widths. Each map row lists the segments of one output tile: runs of datums
// contiguous in both the input tile and the output tile, grouped by input tile so the reader loads each source tile
// once per row and the writer assembles the output tile in L1.
//
// Map entry (uint32): [input_page, input_offset, output_offset, num_elements], offsets in datums within the faced
// tile. Entries with num_elements == 0 are skipped. A sliced map splits each output tile into disjoint byte ranges
// spread over cores; its rows lead with a header entry [output_page, range_start_bytes, range_bytes, 0] and the
// writer stores only that range.
constexpr uint32_t kMappedEntryWords = 4;

// Geometry of a mapped plan; everything the program depends on besides the table contents.
struct TileMappedLayout {
    uint32_t num_rows = 0;
    uint32_t entries_per_page = 0;
    uint32_t num_output_pages = 0;
    bool sliced = false;

    uint32_t page_words() const { return entries_per_page * kMappedEntryWords; }
    uint32_t page_bytes() const { return page_words() * sizeof(uint32_t); }
};

struct TileMappedPlan {
    TileMappedLayout layout;
    // uint32 [num_rows, page_words], one device page per row.
    std::vector<uint32_t> table;
};

// Layout of the mapped plan for this reshape on `num_cores` cores, or nullopt when the map exceeds its size caps or
// its buffers do not fit `l1_budget` even with a single input slot. The sliced map is taken when it pays off and
// fits, else the per-tile map. Only the budget check runs per call; the shape-dependent part is memoized.
std::optional<TileMappedLayout> tile_mapped_layout(
    const ttnn::Shape& in_shape,
    const ttnn::Shape& out_shape,
    uint32_t element_bytes,
    uint32_t tile_bytes,
    uint32_t num_cores,
    uint64_t l1_budget);

// The full plan for a layout tile_mapped_layout returned with the same shapes, sizes and core count.
TileMappedPlan build_tile_mapped_plan(
    const ttnn::Shape& in_shape,
    const ttnn::Shape& out_shape,
    uint32_t element_bytes,
    uint32_t tile_bytes,
    uint32_t num_cores,
    bool sliced);

// Input CB depth that fits `l1_budget` beside one map page and one scratch tile: 8 or 4 enables the pipelined
// reader, 1 the serial one, 0 means nothing fits.
uint32_t tile_mapped_input_depth(const TileMappedLayout& layout, uint32_t tile_bytes, uint64_t l1_budget);

}  // namespace ttnn::operations::data_movement::reshape_codegen
