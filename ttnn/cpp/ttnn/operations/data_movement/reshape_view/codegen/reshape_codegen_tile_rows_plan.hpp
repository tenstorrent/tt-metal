// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Tile-row permutation TILE reshape. With both last dims multiples of 32, every 32-datum row of an output tile is
// one whole row of exactly one input tile, so the reshape permutes tile rows. Output tiles are grouped into batches
// that share source tiles; the device reads each source tile once per batch and assembles its output tiles in L1.
//
// Plan page layout, shared with the kernels through compile-time args:
//   [n_in, n_out, in_page[in_slots], entry[out_slots]]
//   entry = [out_page, n_runs, run[max_runs]]
//   run   = slot << kRunSlotShift | src_row << kRunSrcRowShift | dst_row << kRunDstRowShift | rows
constexpr uint32_t kTileRowsHeaderWords = 2;
constexpr uint32_t kTileRowsEntryHeaderWords = 2;
constexpr uint32_t kRunSlotShift = 20;
constexpr uint32_t kRunSrcRowShift = 15;
constexpr uint32_t kRunDstRowShift = 10;
constexpr uint32_t kRunRowMask = 0x1F;
// Slot that sources a zero tile (output padding rows); also caps the number of real slots.
constexpr uint32_t kZeroSlot = 0xFFF;
constexpr uint32_t kTileRowsWriteGroup = 8;

// Everything the program depends on besides the table contents.
struct TileRowsLayout {
    uint32_t in_slots = 0;
    uint32_t out_slots = 0;
    uint32_t max_runs = 0;
    uint32_t num_batches = 0;
    uint32_t page_words = 0;
    uint32_t num_input_pages = 0;
    uint32_t num_output_pages = 0;

    uint32_t page_bytes() const { return page_words * sizeof(uint32_t); }
    uint32_t entry_words() const { return kTileRowsEntryHeaderWords + max_runs; }
};

struct TileRowsPlan {
    TileRowsLayout layout;
    std::vector<uint32_t> table;  // [num_batches, page_words], one device page per batch
};

// The planner's batches for one shape pair; only build_tile_rows_plan reads them.
struct TileRowsSchedule;

// The planner's verdict for one shape pair.
struct TileRowsChoice {
    TileRowsLayout layout;
    uint32_t max_out = 0;
    // Runs over real rows; source tiles native reads (each output tile reads its distinct sources once); source
    // tiles tile rows reads (each batch reads its distinct sources once).
    uint64_t runs = 0;
    uint64_t native_reads = 0;
    uint64_t batch_reads = 0;
    // Set when this call ran the planner, null when the choice came from the memo.
    std::shared_ptr<const TileRowsSchedule> schedule;
};

// Whether every output tile row is one whole tile row of the input.
bool is_tile_row_permutation(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape);

// nullopt when the shapes are not a tile-row permutation or the plan exceeds its size caps. Memoized on the shapes.
std::optional<TileRowsChoice> plan_tile_rows(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape);

// The plan `choice` describes; reuses its schedule when set.
TileRowsPlan build_tile_rows_plan(
    const ttnn::Shape& in_shape, const ttnn::Shape& out_shape, const TileRowsChoice& choice);

uint64_t tile_rows_l1_bytes(const TileRowsLayout& layout, uint32_t tile_bytes);

}  // namespace ttnn::operations::data_movement::reshape_codegen
