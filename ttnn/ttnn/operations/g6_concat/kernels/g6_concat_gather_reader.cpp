// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// row_gather reader (NCRISC, NOC0): h joins whose offsets are not tile-aligned. Each output page
// is assembled in its cb_output_pages slot from row-range reads of the inputs whose rows land in
// it; only logical input rows are read, each exactly once.
//
// Per block: gather_fetch_block issues every piece's DRAM read, directly into the output slot when
// the source and destination agree modulo the DRAM read alignment, otherwise into the same offset
// (+shift) of a cb_gather_staging mirror slot. gather_place_block then moves the staged pieces to
// their slots with self-aimed L1 -> L1 reads (L1 alignment, always legal).
//
// Raw NoC reads: no kernel_lib helper copies runtime-sized row ranges; local_addr() from
// local_copy_helpers_dataflow.hpp builds the self-aimed source (set_read_state fixes the size at
// compile time, piece sizes here are runtime).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_concat_tile_rows.hpp"

namespace {

using g6_concat::FACE_COLS_PER_TILE;
using g6_concat::FACE_ROWS;
using g6_concat::TILE_ROWS;

// Common runtime args; layout mirrors GATHER_READER_COMMON_ARGS in g6_concat_program_descriptor.py.
constexpr uint32_t CRTA_BLOCK_PAGES = 0;
constexpr uint32_t CRTA_OUT_WT = 1;
constexpr uint32_t CRTA_OUT_HT = 2;
constexpr uint32_t CRTA_OUT_ROWS = 3;
constexpr uint32_t CRTA_NUM_INPUTS = 4;
constexpr uint32_t CRTA_FIRST_INPUT = 5;
constexpr uint32_t CRTA_WORDS_PER_INPUT = 2;  // base address, rows along the joined dim

FORCE_INLINE uint32_t input_base_address(uint32_t input_idx) {
    return get_common_arg_val<uint32_t>(CRTA_FIRST_INPUT + input_idx * CRTA_WORDS_PER_INPUT);
}

FORCE_INLINE uint32_t input_rows(uint32_t input_idx) {
    return get_common_arg_val<uint32_t>(CRTA_FIRST_INPUT + input_idx * CRTA_WORDS_PER_INPUT + 1);
}

FORCE_INLINE uint32_t min3(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t ab = a < b ? a : b;
    return ab < c ? ab : c;
}

struct OutputGeometry {
    uint32_t out_wt;
    uint32_t out_ht;
    uint32_t out_rows;
    uint32_t num_inputs;
};

// Pages [first_col, first_col + num_cols) of output tile row (image, tile_row), held in block slots
// [first_slot, first_slot + num_cols).
struct Segment {
    uint32_t image;
    uint32_t tile_row;
    uint32_t first_col;
    uint32_t num_cols;
    uint32_t first_slot;
};

// Rows [src_row, src_row + num_rows) of an input tile -> rows [dst_row, ...) of an output tile.
// Never crosses row FACE_ROWS on either side, so each face column is one contiguous transfer.
struct RowPiece {
    uint32_t input_idx;
    uint32_t input_row_first_page;  // input page of tile column 0 of the source tile row
    uint32_t src_row;
    uint32_t dst_row;
    uint32_t num_rows;
};

// Run table of one output tile row, generated in order; shared by every column of a segment.
template <typename F>
FORCE_INLINE void for_each_row_piece(const OutputGeometry& geom, const Segment& seg, F&& f) {
    const uint32_t row_lo = seg.tile_row * TILE_ROWS;
    const uint32_t row_hi = (row_lo + TILE_ROWS < geom.out_rows) ? row_lo + TILE_ROWS : geom.out_rows;
    uint32_t input_offset = 0;
    for (uint32_t input_idx = 0; input_idx < geom.num_inputs && input_offset < row_hi; ++input_idx) {
        const uint32_t rows_k = input_rows(input_idx);
        const uint32_t first = row_lo > input_offset ? row_lo : input_offset;
        const uint32_t last = (row_hi < input_offset + rows_k) ? row_hi : input_offset + rows_k;
        if (first < last) {
            const uint32_t input_ht = (rows_k + TILE_ROWS - 1) / TILE_ROWS;
            const uint32_t image_first_page = seg.image * input_ht * geom.out_wt;
            const uint32_t src_end = last - input_offset;
            for (uint32_t src = first - input_offset; src < src_end;) {
                const uint32_t src_row = src % TILE_ROWS;
                const uint32_t dst_row = input_offset + src - row_lo;
                const uint32_t num_rows =
                    min3(FACE_ROWS - src_row % FACE_ROWS, FACE_ROWS - dst_row % FACE_ROWS, src_end - src);
                f(RowPiece{input_idx, image_first_page + (src / TILE_ROWS) * geom.out_wt, src_row, dst_row, num_rows});
                src += num_rows;
            }
        }
        input_offset += rows_k;
    }
}

// Splits block_extent pages starting at (image, tile_row, col) into output-row segments.
struct BlockCursor {
    uint32_t image;
    uint32_t tile_row;
    uint32_t col;

    template <typename F>
    FORCE_INLINE void for_each_segment(const OutputGeometry& geom, uint32_t block_extent, F&& f) const {
        Segment seg{image, tile_row, col, 0, 0};
        uint32_t remaining = block_extent;
        while (remaining > 0) {
            const uint32_t cols_left = geom.out_wt - seg.first_col;
            seg.num_cols = remaining < cols_left ? remaining : cols_left;
            f(seg);
            remaining -= seg.num_cols;
            seg.first_slot += seg.num_cols;
            seg.first_col = 0;
            if (++seg.tile_row == geom.out_ht) {
                seg.tile_row = 0;
                ++seg.image;
            }
        }
    }

    FORCE_INLINE void advance(const OutputGeometry& geom, uint32_t pages) {
        col += pages;
        while (col >= geom.out_wt) {
            col -= geom.out_wt;
            if (++tile_row == geom.out_ht) {
                tile_row = 0;
                ++image;
            }
        }
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather_staging = get_compile_time_arg_val(1);
    constexpr bool has_staging = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(4);
    constexpr auto input_args = TensorAccessorArgs<5>();

    // Inputs are DRAM-interleaved; every piece whose shift is non-zero needs the staging mirror.
    constexpr uint32_t input_read_alignment = NOC_DRAM_READ_ALIGNMENT_BYTES;
    static_assert(has_staging || face_row_bytes % input_read_alignment == 0, "row shifts need cb_gather_staging");
    static_assert(page_bytes % input_read_alignment == 0, "slot addresses must keep the CB base alignment");

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);

    const uint32_t block_pages = get_common_arg_val<uint32_t>(CRTA_BLOCK_PAGES);
    const OutputGeometry geom{
        get_common_arg_val<uint32_t>(CRTA_OUT_WT),
        get_common_arg_val<uint32_t>(CRTA_OUT_HT),
        get_common_arg_val<uint32_t>(CRTA_OUT_ROWS),
        get_common_arg_val<uint32_t>(CRTA_NUM_INPUTS)};

    const uint32_t start_tile_row_global = start_page / geom.out_wt;
    BlockCursor cursor{
        start_tile_row_global / geom.out_ht, start_tile_row_global % geom.out_ht, start_page % geom.out_wt};

    const uint32_t staging_base = has_staging ? get_write_ptr(cb_gather_staging) : 0;
    ASSERT(staging_base % input_read_alignment == 0);

    Noc noc;
    UnicastEndpoint self_endpoint;
    CircularBuffer output_slots(cb_output_pages);

    const uint32_t num_blocks_this_core = (core_pages + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const bool last_block = block_idx + 1 == num_blocks_this_core;
        const uint32_t block_extent = last_block ? core_pages - block_idx * block_pages : block_pages;

        {
            DeviceZoneScopedN("G6C-RESERVE");
            cb_reserve_back(cb_output_pages, block_pages);
        }
        const uint32_t slots_base = get_write_ptr(cb_output_pages);
        ASSERT(slots_base % input_read_alignment == 0);

        {
            DeviceZoneScopedN("G6C-FETCH");
            // gather_fetch_block: piece-major, page-minor, so both offsets stay fixed across a sweep.
            cursor.for_each_segment(geom, block_extent, [&](const Segment& seg) {
                for_each_row_piece(geom, seg, [&](const RowPiece& piece) {
                    const auto input = TensorAccessor(input_args, input_base_address(piece.input_idx), page_bytes);
                    const uint32_t piece_bytes = piece.num_rows * face_row_bytes;
                    for (uint32_t face_col = 0; face_col < FACE_COLS_PER_TILE; ++face_col) {
                        const uint32_t src_off = g6_concat::face_offset<face_row_bytes>(piece.src_row, face_col);
                        const uint32_t dst_off = g6_concat::face_offset<face_row_bytes>(piece.dst_row, face_col);
                        const uint32_t shift = (src_off - dst_off) % input_read_alignment;
                        uint32_t dst_addr =
                            (shift == 0 ? slots_base : staging_base) + seg.first_slot * page_bytes + dst_off + shift;
                        uint32_t input_page = piece.input_row_first_page + seg.first_col;
                        for (uint32_t col = 0; col < seg.num_cols; ++col) {
                            noc_async_read(input.get_noc_addr(input_page, src_off), dst_addr, piece_bytes);
                            ++input_page;
                            dst_addr += page_bytes;
                        }
                    }
                });
            });
            noc_async_read_barrier();
        }

        if constexpr (has_staging) {
            DeviceZoneScopedN("G6C-PLACE");
            // gather_place_block: staged pieces -> their output slots, same block, same thread.
            cursor.for_each_segment(geom, block_extent, [&](const Segment& seg) {
                for_each_row_piece(geom, seg, [&](const RowPiece& piece) {
                    const uint32_t piece_bytes = piece.num_rows * face_row_bytes;
                    for (uint32_t face_col = 0; face_col < FACE_COLS_PER_TILE; ++face_col) {
                        const uint32_t src_off = g6_concat::face_offset<face_row_bytes>(piece.src_row, face_col);
                        const uint32_t dst_off = g6_concat::face_offset<face_row_bytes>(piece.dst_row, face_col);
                        const uint32_t shift = (src_off - dst_off) % input_read_alignment;
                        if (shift == 0) {
                            continue;
                        }
                        uint32_t slot_offset = seg.first_slot * page_bytes + dst_off;
                        uint32_t staged_addr = staging_base + slot_offset + shift;
                        for (uint32_t col = 0; col < seg.num_cols; ++col) {
                            noc.async_read(
                                self_endpoint,
                                output_slots,
                                piece_bytes,
                                dataflow_kernel_lib::local_addr(staged_addr, noc.get_noc_id()),
                                {.offset_bytes = slot_offset});
                            slot_offset += page_bytes;
                            staged_addr += page_bytes;
                        }
                    }
                });
            });
            noc_async_read_barrier();
        }

        // Nominal count on the ragged last block keeps the ring cycle closed.
        cb_push_back(cb_output_pages, block_pages);
        cursor.advance(geom, block_extent);
    }
}
