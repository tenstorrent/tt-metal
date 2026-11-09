// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace ttnn::operations::data_movement::reshape_new {

// How a reshape from one TensorSpec to another is carried out. Decided from the two specs alone.
//   NoOp     -- same shapes and a functionally identical memory config: the input is the result.
//   View     -- the input's pages are, byte for byte and in order, the output's pages, and the output keeps the
//               input's distribution: a zero-copy relabeling of the input buffer.
//   PageCopy -- the page sequences are identical but the distribution differs (interleaved <-> sharded, another
//               shard grid, DRAM <-> L1): output page i is input page i.
//   Gather   -- everything else: each output page is assembled from element runs of the input.
enum class ReshapeKind : uint8_t { NoOp, View, PageCopy, Gather };

ReshapeKind plan_reshape(const tt::tt_metal::TensorSpec& input, const tt::tt_metal::TensorSpec& output);

// True when output page i holds exactly the bytes of input page i, for every page: same dtype, layout and tile,
// same last dim and padded last dim, same page size and physical 2D shape, and the same row structure (the same
// dim -2 and its padding, or no row padding on either side).
bool page_sequences_identical(const tt::tt_metal::TensorSpec& input, const tt::tt_metal::TensorSpec& output);

// One side of a reshape as the Gather kernels walk it. A tensor is viewed as `batch` x `height` x `width` logical
// elements (dims -2 and -1, everything before folded into the batch); its pages form a grid of rows of
// `pages_per_row` pages, with `padded_height` rows (dim -2 padded) per batch.
//   ROW_MAJOR: one page row per element row; a page holds `page_width` elements of it.
//   TILE:      one page row per tile row; a page is one tile, `page_width` is the tile width.
struct ReshapeSideGeometry {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t padded_height = 0;
    uint32_t batch = 0;
    uint32_t page_width = 0;
    uint32_t pages_per_row = 0;
    uint32_t num_pages = 0;
};

// Fatal when `spec` is outside what the kernels address: padding outside dims -2 and -1, or a volume (logical or
// padded) that does not fit in 32 bits.
ReshapeSideGeometry side_geometry(const tt::tt_metal::TensorSpec& spec);

// Everything the PageCopy program needs, from the specs and the two buffers' alignments. The work unit is one
// chunk of one page: pages wider than the chunk size are moved in pieces, so the L1 footprint stays bounded.
struct PageCopyParams {
    uint32_t num_pages = 0;
    uint32_t page_bytes = 0;
    uint32_t chunk_bytes = 0;  // a multiple of both buffers' alignment, except when it is the whole page
    uint32_t chunks_per_page = 1;
    uint32_t num_units = 0;
    uint32_t alignment = 0;        // the larger buffer alignment; chunks are staged on this boundary of L1
    uint32_t entry_bytes = 0;      // DFB entry: a chunk rounded up to `alignment`, plus `alignment` of slack
    uint32_t units_per_batch = 0;  // units moved per barrier
    uint32_t num_entries = 0;      // two batches in flight
};

PageCopyParams make_page_copy_params(
    const tt::tt_metal::TensorSpec& input,
    const tt::tt_metal::TensorSpec& output,
    uint32_t input_alignment,
    uint32_t output_alignment);

// Everything the Gather program needs. The work unit is one output tile (TILE), one chunk of one output page
// (ROW_MAJOR, where wide pages are split so the staging footprint stays bounded), or a run of `pages_per_unit`
// consecutive small ROW_MAJOR pages. The reader stages the input fragments of a unit into stage entries of a
// fixed size, so the L1 footprint does not depend on the shapes.
struct GatherParams {
    ReshapeSideGeometry in;
    ReshapeSideGeometry out;
    bool tiled = false;
    uint32_t element_size = 0;
    uint32_t tile_height = 1;
    uint32_t tile_width = 1;
    uint32_t face_height = 1;
    uint32_t face_width = 1;
    uint32_t page_bytes = 0;       // one output page
    uint32_t unit_bytes = 0;       // bytes written per page of a unit: a tile, a ROW_MAJOR chunk, or a page
    uint32_t chunks_per_page = 1;  // ROW_MAJOR units per output page
    uint32_t pages_per_unit = 1;   // ROW_MAJOR pages per unit (1 unless pages are small)
    uint32_t page_slot_bytes = 0;  // assembly stride between the pages of a unit (unit_bytes, NOC-word aligned)
    uint32_t unit_slot_bytes = 0;  // assembly footprint of one unit
    uint32_t num_units = 0;
    uint32_t source_alignment = 0;  // reads start on this boundary of the input buffer, and land on it in L1
    uint32_t records_per_entry = 0;
    uint32_t header_bytes = 0;       // a multiple of source_alignment
    uint32_t stage_data_bytes = 0;   // a multiple of source_alignment
    uint32_t stage_entry_bytes = 0;  // header, data, and one source_alignment of slack
    uint32_t num_stage_entries = 0;  // per stage buffer; the reader alternates between two
};

// Number of uint32 words before the first fragment record in a stage entry, and words per record. Shared with
// the kernels through compile-time arguments.
inline constexpr uint32_t kGatherHeaderWords = 6;
inline constexpr uint32_t kGatherRecordWords = 3;

// `num_cores` is the number of cores the units are split over; it bounds how many small pages share a unit.
GatherParams make_gather_params(
    const tt::tt_metal::TensorSpec& input,
    const tt::tt_metal::TensorSpec& output,
    uint32_t input_alignment,
    uint32_t num_cores);

}  // namespace ttnn::operations::data_movement::reshape_new
