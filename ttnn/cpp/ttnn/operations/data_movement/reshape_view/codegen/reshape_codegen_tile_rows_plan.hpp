// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

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

// Whether every output tile row is one whole tile row of the input.
bool is_tile_row_permutation(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape);

// Layout of the plan the factory would upload, or nullopt when the shapes are not a tile-row permutation or exceed
// the caps. Memoized on the shapes: the gate and program hash run per call and planning walks every output tile row.
std::optional<TileRowsLayout> tile_rows_layout(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape);

// The full plan whose layout tile_rows_layout returned; only call when that returned a value.
TileRowsPlan build_tile_rows_plan(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape);

uint64_t tile_rows_l1_bytes(const TileRowsLayout& layout, uint32_t tile_bytes);

// Whether the tile-rows program measures faster than native's tiled reshape on `num_cores` cores. Native reads every
// distinct source tile of each output tile; tile rows reads a source once per batch of output tiles but runs fewer
// batches than native runs tiles at once. It wins when batching cuts reads by at least the smallest batch size, or
// when every output tile is whole face halves of its sources and native needs more than one round. False when the
// shapes are not a tile-row permutation.
bool tile_rows_beats_native(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape, uint32_t num_cores);

// Whether the tile-rows program's buffers fit `l1_budget`; false when the shapes are not a tile-row permutation or
// exceed the caps.
bool tile_rows_fits(const Tensor& input, const ttnn::Shape& logical_output_shape, uint64_t l1_budget);

}  // namespace ttnn::operations::data_movement::reshape_codegen
