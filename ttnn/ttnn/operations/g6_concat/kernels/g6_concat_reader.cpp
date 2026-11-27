// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_concat reader (NCRISC, NOC0). Fills cb_output_pages one block of `block_pages` consecutive
// output pages at a time, in one of two regimes fixed at program build (CT arg `regime`, so each
// regime compiles to its own binary):
//   page_copy  — each output page is one input page (whole, or its valid rows when trimmed);
//   row_gather — each output page is assembled from row-range reads of the inputs whose rows land
//                in it (h join with an input starting inside a tile).
//
// Raw dataflow API throughout: no kernel_lib helper copies pages or row ranges between buffers.
// set_read_state/read_with_state (local_copy_helpers_dataflow.hpp) bake the transfer size into a
// template parameter, while row pieces are runtime-sized, so only its local_addr() source builder
// is used for the place phase.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_concat_tile_rows.hpp"

constexpr uint32_t REGIME_PAGE_COPY = 0;
constexpr uint32_t REGIME_ROW_GATHER = 1;

constexpr uint32_t regime = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_pages = get_compile_time_arg_val(1);
constexpr uint32_t cb_gather_staging = get_compile_time_arg_val(2);
constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t row_bytes = get_compile_time_arg_val(4);
constexpr uint32_t trim_rows_max = get_compile_time_arg_val(5);
constexpr bool has_staging = get_compile_time_arg_val(6) != 0;
constexpr uint32_t input_read_alignment = get_compile_time_arg_val(7);
constexpr auto input_args = TensorAccessorArgs<8>();

constexpr uint32_t face_bytes = g6c::FACE_ROWS * row_bytes;

static_assert(page_bytes <= NOC_MAX_BURST_SIZE, "every transfer is issued as a single NoC packet");
static_assert((input_read_alignment & (input_read_alignment - 1)) == 0, "read alignment must be a power of 2");
static_assert(input_read_alignment == NOC_DRAM_READ_ALIGNMENT_BYTES, "host and device disagree on DRAM read alignment");
// A whole face half moves as one span that is never staged; its offsets differ by whole faces.
static_assert(face_bytes % input_read_alignment == 0, "face size must be a multiple of the read alignment");

// Common runtime args: [K, block_pages, out_Wt, out_Ht, v_last, (addr_k, extent_k) * K]
// extent_k is C_k (pages of input k per outer slice) in page_copy and n_k (rows) in row_gather.
constexpr uint32_t CARG_NUM_INPUTS = 0;
constexpr uint32_t CARG_BLOCK_PAGES = 1;
constexpr uint32_t CARG_OUT_WT = 2;
constexpr uint32_t CARG_OUT_HT = 3;
constexpr uint32_t CARG_LAST_VALID_ROWS = 4;
constexpr uint32_t CARG_INPUTS = 5;

FORCE_INLINE uint32_t input_addr(uint32_t k) { return get_common_arg_val<uint32_t>(CARG_INPUTS + 2 * k); }
FORCE_INLINE uint32_t input_extent(uint32_t k) { return get_common_arg_val<uint32_t>(CARG_INPUTS + 2 * k + 1); }

FORCE_INLINE auto input_accessor(uint32_t k) { return TensorAccessor(input_args, input_addr(k), page_bytes); }

// ---------------------------------------------------------------------------------------------
// page_copy
// ---------------------------------------------------------------------------------------------

// Output page o*C + Coff_k + c' is input k's page o*C_k + c'.
struct PageCopyCursor {
    uint32_t slice;
    uint32_t input;
    uint32_t page_in_slice;
    uint32_t num_inputs;
};

template <typename Accessor>
FORCE_INLINE void read_page(const Accessor& acc, uint32_t page, uint32_t slot, uint32_t valid_rows) {
    if (g6c::trimmed<trim_rows_max>(valid_rows)) {
        // Same in-tile offset on both sides: always alignment-legal.
        g6c::for_each_valid_row_span(valid_rows, row_bytes, [&](uint32_t offset, uint32_t bytes) {
            noc_async_read<NOC_MAX_BURST_SIZE>(acc.get_noc_addr(page, offset), slot + offset, bytes);
        });
    } else {
        noc_async_read<NOC_MAX_BURST_SIZE>(acc.get_noc_addr(page), slot, page_bytes);
    }
}

void page_fetch_block(uint32_t extent, PageCopyCursor& cur, g6c::OutputRowCursor& rows, uint32_t slot) {
    uint32_t remaining = extent;
    while (remaining > 0) {
        const uint32_t pages_per_slice = input_extent(cur.input);
        uint32_t run = pages_per_slice - cur.page_in_slice;
        if (run > remaining) {
            run = remaining;
        }
        const auto acc = input_accessor(cur.input);
        uint32_t page = cur.slice * pages_per_slice + cur.page_in_slice;
        for (uint32_t i = 0; i < run; ++i) {
            read_page(acc, page, slot, rows.valid_rows());
            rows.advance();
            ++page;
            slot += page_bytes;
        }
        cur.page_in_slice += run;
        if (cur.page_in_slice == pages_per_slice) {
            cur.page_in_slice = 0;
            if (++cur.input == cur.num_inputs) {
                cur.input = 0;
                ++cur.slice;
            }
        }
        remaining -= run;
    }
    noc_async_read_barrier();
}

// ---------------------------------------------------------------------------------------------
// row_gather
// ---------------------------------------------------------------------------------------------

struct GatherGeometry {
    uint32_t num_inputs;
    uint32_t tiles_per_row;
    uint32_t tile_rows;
    uint32_t last_valid_rows;
};

// Position of the next output page: image, tile row, tile column.
struct GatherCursor {
    uint32_t image;
    uint32_t tile_row;
    uint32_t col;
};

// Visits the run table of output tile row `tile_row`: every maximal row range of one input that
// lands in it, split so no piece crosses row 16 (face half) on the source or the destination.
// fn(k, input_tile_row, src_row, dst_row, num_rows). Only logical input rows are visited, each once.
template <typename Fn>
FORCE_INLINE void for_each_row_piece(uint32_t tile_row, const GatherGeometry& geo, Fn&& fn) {
    const uint32_t row_lo = tile_row * g6c::TILE_ROWS;
    const uint32_t row_hi = row_lo + (tile_row == geo.tile_rows - 1 ? geo.last_valid_rows : g6c::TILE_ROWS);
    uint32_t input_offset = 0;
    for (uint32_t k = 0; k < geo.num_inputs && input_offset < row_hi; ++k) {
        const uint32_t input_end = input_offset + input_extent(k);
        if (input_end > row_lo) {
            uint32_t out_row = input_offset > row_lo ? input_offset : row_lo;
            const uint32_t out_end = input_end < row_hi ? input_end : row_hi;
            uint32_t in_row = out_row - input_offset;
            while (out_row < out_end) {
                const uint32_t src_row = in_row % g6c::TILE_ROWS;
                const uint32_t dst_row = out_row - row_lo;
                uint32_t num_rows = out_end - out_row;
                const uint32_t src_room = g6c::FACE_ROWS - src_row % g6c::FACE_ROWS;
                const uint32_t dst_room = g6c::FACE_ROWS - dst_row % g6c::FACE_ROWS;
                num_rows = num_rows < src_room ? num_rows : src_room;
                num_rows = num_rows < dst_room ? num_rows : dst_room;
                fn(k, in_row / g6c::TILE_ROWS, src_row, dst_row, num_rows);
                out_row += num_rows;
                in_row += num_rows;
            }
        }
        input_offset = input_end;
    }
}

// A whole face half (16 rows starting on row 0 or 16 on both sides) is one contiguous span across
// both face columns; anything shorter is one span per face column.
FORCE_INLINE bool whole_face_half(uint32_t num_rows) { return num_rows == g6c::FACE_ROWS; }

// Phase A for one output-row segment: columns [col_begin, col_begin + num_cols) of one tile row.
// Piece-major, page-minor: offsets and the direct/staged choice are fixed across the page sweep.
// A piece whose source and destination differ mod the read alignment lands in the staging mirror
// slot at dst_off + shift so the DRAM read is legal; phase B moves it into place.
void gather_fetch_segment(
    const GatherCursor& seg, uint32_t num_cols, const GatherGeometry& geo, uint32_t out_slot, uint32_t staging_slot) {
    for_each_row_piece(
        seg.tile_row,
        geo,
        [&](uint32_t k, uint32_t input_tile_row, uint32_t src_row, uint32_t dst_row, uint32_t num_rows) {
            const auto acc = input_accessor(k);
            const uint32_t input_tile_rows = (input_extent(k) + g6c::TILE_ROWS - 1) / g6c::TILE_ROWS;
            const uint32_t first_page = (seg.image * input_tile_rows + input_tile_row) * geo.tiles_per_row + seg.col;
            const bool merged = whole_face_half(num_rows);
            const uint32_t num_face_cols = merged ? 1 : 2;
            const uint32_t bytes = merged ? 2 * face_bytes : num_rows * row_bytes;
            for (uint32_t face_col = 0; face_col < num_face_cols; ++face_col) {
                const uint32_t src_off = g6c::face_row_offset(src_row, face_col, row_bytes);
                const uint32_t dst_off = g6c::face_row_offset(dst_row, face_col, row_bytes);
                const uint32_t shift = (src_off - dst_off) & (input_read_alignment - 1);
                uint32_t dst = out_slot + dst_off;
                if constexpr (has_staging) {
                    if (shift != 0) {
                        dst = staging_slot + dst_off + shift;
                    }
                } else {
                    ASSERT(shift == 0);
                }
                for (uint32_t c = 0; c < num_cols; ++c) {
                    noc_async_read<NOC_MAX_BURST_SIZE>(acc.get_noc_addr(first_page + c, src_off), dst, bytes);
                    dst += page_bytes;
                }
            }
        });
}

// Phase B for one output-row segment: self-aimed L1->L1 reads (16 B alignment, always legal) from
// the staging mirror into the output slots, for exactly the pieces phase A staged.
void gather_place_segment(
    const GatherCursor& seg, uint32_t num_cols, const GatherGeometry& geo, uint32_t out_slot, uint32_t staging_slot) {
    for_each_row_piece(
        seg.tile_row, geo, [&](uint32_t, uint32_t, uint32_t src_row, uint32_t dst_row, uint32_t num_rows) {
            if (whole_face_half(num_rows)) {
                return;
            }
            const uint32_t bytes = num_rows * row_bytes;
            for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                const uint32_t src_off = g6c::face_row_offset(src_row, face_col, row_bytes);
                const uint32_t dst_off = g6c::face_row_offset(dst_row, face_col, row_bytes);
                const uint32_t shift = (src_off - dst_off) & (input_read_alignment - 1);
                if (shift == 0) {
                    continue;
                }
                const auto self = dataflow_kernel_lib::local_addr(staging_slot + dst_off + shift, noc_index);
                uint64_t src = get_noc_addr(self.noc_x, self.noc_y, self.addr, noc_index);
                uint32_t dst = out_slot + dst_off;
                for (uint32_t c = 0; c < num_cols; ++c) {
                    noc_async_read<NOC_MAX_BURST_SIZE>(src, dst, bytes);
                    src += page_bytes;
                    dst += page_bytes;
                }
            }
        });
}

// Walks the output-row segments of one block — `extent` pages starting at `cur` — calling
// fn(segment, num_cols, slot_offset_bytes). Returns the cursor after the block.
template <typename Fn>
FORCE_INLINE GatherCursor for_each_segment(GatherCursor seg, uint32_t extent, const GatherGeometry& geo, Fn&& fn) {
    uint32_t slot_offset = 0;
    while (extent > 0) {
        uint32_t num_cols = geo.tiles_per_row - seg.col;
        if (num_cols > extent) {
            num_cols = extent;
        }
        fn(seg, num_cols, slot_offset);
        slot_offset += num_cols * page_bytes;
        extent -= num_cols;
        seg.col += num_cols;
        if (seg.col == geo.tiles_per_row) {
            seg.col = 0;
            if (++seg.tile_row == geo.tile_rows) {
                seg.tile_row = 0;
                ++seg.image;
            }
        }
    }
    return seg;
}

FORCE_INLINE GatherCursor gather_fetch_block(
    const GatherCursor& cur, uint32_t extent, const GatherGeometry& geo, uint32_t out_slot, uint32_t staging_slot) {
    const GatherCursor next =
        for_each_segment(cur, extent, geo, [&](const GatherCursor& seg, uint32_t num_cols, uint32_t slot_offset) {
            gather_fetch_segment(seg, num_cols, geo, out_slot + slot_offset, staging_slot + slot_offset);
        });
    noc_async_read_barrier();
    return next;
}

FORCE_INLINE void gather_place_block(
    const GatherCursor& cur, uint32_t extent, const GatherGeometry& geo, uint32_t out_slot, uint32_t staging_slot) {
    for_each_segment(cur, extent, geo, [&](const GatherCursor& seg, uint32_t num_cols, uint32_t slot_offset) {
        gather_place_segment(seg, num_cols, geo, out_slot + slot_offset, staging_slot + slot_offset);
    });
    noc_async_read_barrier();
}

// ---------------------------------------------------------------------------------------------

void kernel_main() {
    const uint32_t num_inputs = get_common_arg_val<uint32_t>(CARG_NUM_INPUTS);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(CARG_BLOCK_PAGES);
    const uint32_t out_Wt = get_common_arg_val<uint32_t>(CARG_OUT_WT);
    const uint32_t out_Ht = get_common_arg_val<uint32_t>(CARG_OUT_HT);
    const uint32_t last_valid_rows = get_common_arg_val<uint32_t>(CARG_LAST_VALID_ROWS);

    // Per-core runtime args: [pages, start tile_row, start col, regime cursor...]
    const uint32_t pages_this_core = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t start_col = get_arg_val<uint32_t>(2);

    ASSERT(get_write_ptr(cb_output_pages) % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);

    if constexpr (regime == REGIME_PAGE_COPY) {
        PageCopyCursor cur{
            .slice = get_arg_val<uint32_t>(3),
            .input = get_arg_val<uint32_t>(4),
            .page_in_slice = get_arg_val<uint32_t>(5),
            .num_inputs = num_inputs};
        g6c::OutputRowCursor rows{
            .tile_row = start_tile_row,
            .col = start_col,
            .tiles_per_row = out_Wt,
            .tile_rows = out_Ht,
            .last_valid_rows = last_valid_rows};

        uint32_t remaining = pages_this_core;
        while (remaining > 0) {
            const uint32_t extent = remaining < block_pages ? remaining : block_pages;
            {
                DeviceZoneScopedN("G6C-RESERVE");
                cb_reserve_back(cb_output_pages, block_pages);
            }
            {
                DeviceZoneScopedN("G6C-FETCH");
                page_fetch_block(extent, cur, rows, get_write_ptr(cb_output_pages));
            }
            // Nominal count even on the ragged last block: keeps the ring in step with the writer.
            cb_push_back(cb_output_pages, block_pages);
            remaining -= extent;
        }
    } else {
        const GatherGeometry geo{
            .num_inputs = num_inputs, .tiles_per_row = out_Wt, .tile_rows = out_Ht, .last_valid_rows = last_valid_rows};
        GatherCursor cur{.image = get_arg_val<uint32_t>(3), .tile_row = start_tile_row, .col = start_col};
        uint32_t staging_slot = 0;
        if constexpr (has_staging) {
            staging_slot = get_write_ptr(cb_gather_staging);
            ASSERT(staging_slot % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);
        }

        uint32_t remaining = pages_this_core;
        while (remaining > 0) {
            const uint32_t extent = remaining < block_pages ? remaining : block_pages;
            {
                DeviceZoneScopedN("G6C-RESERVE");
                cb_reserve_back(cb_output_pages, block_pages);
            }
            const uint32_t out_slot = get_write_ptr(cb_output_pages);
            GatherCursor next;
            {
                DeviceZoneScopedN("G6C-FETCH");
                next = gather_fetch_block(cur, extent, geo, out_slot, staging_slot);
            }
            if constexpr (has_staging) {
                DeviceZoneScopedN("G6C-PLACE");
                gather_place_block(cur, extent, geo, out_slot, staging_slot);
            }
            cb_push_back(cb_output_pages, block_pages);
            cur = next;
            remaining -= extent;
        }
    }
}
