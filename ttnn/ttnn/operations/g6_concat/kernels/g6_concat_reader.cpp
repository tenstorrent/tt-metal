// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_concat reader (NCRISC, NoC0). Fills cb_moved_pages one block at a time.
//
// page_copy:  gather_pages_block — every output page has one source page; read its
//             valid-row prefix straight into the slot.
// row_gather: rows of several inputs share an output tile. Per block (one output tile row
//             x `extent` columns) the run list is derived on the fly and moved in three
//             barriered phases: off-parity runs DRAM -> stage, one span copy stage -> slot,
//             on-parity runs DRAM -> slot. The span copy also lands stage garbage on
//             on-parity rows (fixed by the later direct reads) and on padding rows (never
//             written out), so the order of the three phases is load-bearing.
//
// Raw NoC + TensorAccessor: no kernel_lib dataflow helper copies whole or partial pages
// (op_design.md, Helpers considered and rejected).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_concat_pages.hpp"

namespace {

// Common runtime-arg table: header, then (base address, count) per input. count is the
// input's page-matrix segment width (page_copy) or its row count along h (row_gather).
constexpr uint32_t ARG_NUM_INPUTS = 0;
constexpr uint32_t ARG_PM_COLS = 1;
constexpr uint32_t ARG_OUT_TILE_COLS = 2;
constexpr uint32_t ARG_OUT_TILE_ROWS_PER_IMAGE = 3;
constexpr uint32_t ARG_OUT_ROWS_TAIL = 4;
constexpr uint32_t HEADER_WORDS = 5;
constexpr uint32_t WORDS_PER_INPUT = 2;

FORCE_INLINE uint32_t input_addr(uint32_t k) {
    return get_common_arg_val<uint32_t>(HEADER_WORDS + WORDS_PER_INPUT * k);
}

FORCE_INLINE uint32_t input_count(uint32_t k) {
    return get_common_arg_val<uint32_t>(HEADER_WORDS + WORDS_PER_INPUT * k + 1);
}

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
FORCE_INLINE uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

// page_copy source of an output page. The output's pages form a matrix of pm_cols columns
// per row; each row is K contiguous segments, segment k being seg_pages[k] pages of input k.
// Column c of row a in segment k is page a * seg_pages[k] + (c - seg_start) of input k.
struct SourceCursor {
    uint32_t pm_cols;
    uint32_t row;
    uint32_t col;
    uint32_t input;
    uint32_t seg_start;
    uint32_t seg_pages;
    uint32_t src_page;

    SourceCursor(uint32_t first_page, uint32_t pm_cols_) : pm_cols(pm_cols_) {
        row = first_page / pm_cols;
        col = first_page - row * pm_cols;
        input = 0;
        seg_start = 0;
        seg_pages = input_count(0);
        while (col >= seg_start + seg_pages) {
            seg_start += seg_pages;
            seg_pages = input_count(++input);
        }
        src_page = row * seg_pages + (col - seg_start);
    }

    void advance() {
        ++col;
        ++src_page;
        if (col == seg_start + seg_pages) {
            if (col == pm_cols) {
                col = 0;
                ++row;
                input = 0;
                seg_start = 0;
            } else {
                seg_start = col;
                ++input;
            }
            seg_pages = input_count(input);
            src_page = row * seg_pages;
        }
    }
};

// Calls fn(input, src_tile_row, src_row, dst_row, rows) for each run piece of output rows
// [row_lo, row_lo + valid_rows) of image `image`. A piece is rows of one input inside one
// source tile row and inside one face band (rows 0-15 or 16-31) on both sides. Only an
// input's logical rows are visited: its tile padding never reaches an output row.
template <typename Fn>
FORCE_INLINE void for_each_run_piece(
    uint32_t num_inputs, uint32_t image, uint32_t row_lo, uint32_t valid_rows, Fn&& fn) {
    const uint32_t row_hi = row_lo + valid_rows;
    uint32_t offset = 0;
    for (uint32_t k = 0; k < num_inputs && offset < row_hi; ++k) {
        const uint32_t rows_k = input_count(k);
        const uint32_t lo = max_u32(row_lo, offset);
        const uint32_t hi = min_u32(row_hi, offset + rows_k);
        if (lo < hi) {
            const uint32_t tile_rows_k = (rows_k + g6c::TILE_ROWS - 1) / g6c::TILE_ROWS;
            for (uint32_t r = lo; r < hi;) {
                const uint32_t j = r - offset;
                const uint32_t src_row = j % g6c::TILE_ROWS;
                const uint32_t dst_row = r - row_lo;
                uint32_t rows = hi - r;
                rows = min_u32(rows, g6c::FACE_ROWS - (src_row % g6c::FACE_ROWS));
                rows = min_u32(rows, g6c::FACE_ROWS - (dst_row % g6c::FACE_ROWS));
                fn(k, image * tile_rows_k + j / g6c::TILE_ROWS, src_row, dst_row, rows);
                r += rows;
            }
        }
        offset += rows_k;
    }
}

// Byte layout of one run piece inside a tile: the left-face run and, at +face_bytes, the
// right-face run. A full band (16 rows) is the two faces back to back: one transfer.
template <uint32_t face_row_bytes>
struct PieceLayout {
    static constexpr uint32_t face_bytes = g6c::FACE_ROWS * face_row_bytes;
    uint32_t src_off;
    uint32_t dst_off;
    uint32_t bytes;
    bool full_band;

    PieceLayout(uint32_t src_row, uint32_t dst_row, uint32_t rows) {
        src_off = (src_row / g6c::FACE_ROWS) * 2 * face_bytes + (src_row % g6c::FACE_ROWS) * face_row_bytes;
        dst_off = (dst_row / g6c::FACE_ROWS) * 2 * face_bytes + (dst_row % g6c::FACE_ROWS) * face_row_bytes;
        full_band = rows == g6c::FACE_ROWS;
        bytes = full_band ? 2 * face_bytes : rows * face_row_bytes;
    }

    // End of the bytes this piece covers in a tile, relative to the tile base.
    uint32_t dst_end() const { return full_band ? dst_off + bytes : dst_off + face_bytes + bytes; }

    template <uint32_t page_bytes, typename Accessor>
    FORCE_INLINE void read(const Accessor& src, uint32_t src_page, uint32_t dst_tile_addr) const {
        noc_async_read<page_bytes>(src.get_noc_addr(src_page, src_off), dst_tile_addr + dst_off, bytes);
        if (!full_band) {
            noc_async_read<page_bytes>(
                src.get_noc_addr(src_page, src_off + face_bytes), dst_tile_addr + dst_off + face_bytes, bytes);
        }
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_moved_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_parity_stage = get_compile_time_arg_val(1);
    constexpr uint32_t block_pages = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t dram_read_align = get_compile_time_arg_val(5);
    constexpr bool is_row_gather = get_compile_time_arg_val(6) != 0;
    constexpr bool has_stage = get_compile_time_arg_val(7) != 0;
    constexpr auto in_args = TensorAccessorArgs<8>();

    static_assert(dram_read_align == NOC_DRAM_READ_ALIGNMENT_BYTES, "host and NoC disagree on DRAM-read alignment");
    static_assert(page_bytes % dram_read_align == 0, "every slot and stage tile must share one parity class");

    // row_gather: DRAM reads need (L1 dst % A) == (DRAM src % A). A run whose source and
    // destination rows differ by an odd count is `stage_class` off; it lands in the stage at a
    // base shifted by that class and reaches the slot by an L1->L1 copy (16 B congruence only).
    constexpr uint32_t stage_class = face_row_bytes % dram_read_align;
    static_assert(!is_row_gather || has_stage == (stage_class != 0), "host and kernel disagree on the parity stage");
    static_assert(stage_class == 0 || 2 * stage_class == dram_read_align, "one parity stage covers one class only");
    static_assert(stage_class % NOC_L1_READ_ALIGNMENT_BYTES == 0, "stage->slot copy needs L1 congruence");
    static_assert((g6c::FACE_ROWS * face_row_bytes) % dram_read_align == 0, "both face halves of a run share a class");

    const uint32_t num_inputs = get_common_arg_val<uint32_t>(ARG_NUM_INPUTS);
    const uint32_t pm_cols = get_common_arg_val<uint32_t>(ARG_PM_COLS);
    const uint32_t out_tile_cols = get_common_arg_val<uint32_t>(ARG_OUT_TILE_COLS);
    const uint32_t out_tile_rows_per_image = get_common_arg_val<uint32_t>(ARG_OUT_TILE_ROWS_PER_IMAGE);
    const uint32_t out_rows_tail = get_common_arg_val<uint32_t>(ARG_OUT_ROWS_TAIL);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    g6c::OutPageCursor cursor(first_page, out_tile_cols, out_tile_rows_per_image, out_rows_tail);

    if constexpr (!is_row_gather) {
        SourceCursor source(first_page, pm_cols);
        uint32_t pages_done = 0;
        while (pages_done < num_pages) {
            const uint32_t extent = g6c::block_extent<block_pages, false>(num_pages - pages_done, cursor);
            {
                DeviceZoneScopedN("g6c_r_reserve_back");
                cb_reserve_back(cb_moved_pages, block_pages);
            }
            {
                DeviceZoneScopedN("g6c_gather_pages_block");
                uint32_t slot_addr = get_write_ptr(cb_moved_pages);
                for (uint32_t i = 0; i < extent; ++i) {
                    const auto src = TensorAccessor(in_args, input_addr(source.input), page_bytes);
                    const uint32_t src_page = source.src_page;
                    // Source and output page have the same valid rows: w and outer keep H, and
                    // h page_copy has tile-aligned offsets, so only the last input's tail is partial.
                    g6c::for_each_valid_row_piece<page_bytes, face_row_bytes>(
                        cursor.valid_rows(), [&](uint32_t offset, uint32_t bytes) {
                            noc_async_read<page_bytes>(src.get_noc_addr(src_page, offset), slot_addr + offset, bytes);
                        });
                    slot_addr += page_bytes;
                    cursor.advance();
                    source.advance();
                }
            }
            {
                DeviceZoneScopedN("g6c_r_read_barrier");
                noc_async_read_barrier();
            }
            // Nominal push: a ragged block narrows only the transfers, never the ring step.
            cb_push_back(cb_moved_pages, block_pages);
            pages_done += extent;
        }
    } else {
        using Piece = PieceLayout<face_row_bytes>;

        auto parity_class = [](uint64_t src_noc_addr, uint32_t dst_l1_addr) -> uint32_t {
            return (static_cast<uint32_t>(src_noc_addr) - dst_l1_addr) & (dram_read_align - 1);
        };

        uint32_t pages_done = 0;
        while (pages_done < num_pages) {
            const uint32_t extent = g6c::block_extent<block_pages, true>(num_pages - pages_done, cursor);
            const uint32_t image = cursor.image;
            const uint32_t row_lo = cursor.tile_row * g6c::TILE_ROWS;
            const uint32_t valid_rows = cursor.valid_rows();
            const uint32_t first_col = cursor.col;
            {
                DeviceZoneScopedN("g6c_r_reserve_back");
                cb_reserve_back(cb_moved_pages, block_pages);
            }
            const uint32_t slot_base = get_write_ptr(cb_moved_pages);

            if constexpr (has_stage) {
                const uint32_t stage_cb_addr = get_write_ptr(cb_parity_stage);
                const uint32_t stage_base =
                    stage_cb_addr + ((slot_base + stage_class - stage_cb_addr) & (dram_read_align - 1));
                // In-tile byte range [span_lo, span_hi) holding every off-parity row of the tile row.
                uint32_t span_lo = page_bytes;
                uint32_t span_hi = 0;
                {
                    DeviceZoneScopedN("g6c_stage_offparity_rows_block");
                    for_each_run_piece(
                        num_inputs,
                        image,
                        row_lo,
                        valid_rows,
                        [&](uint32_t k, uint32_t src_tile_row, uint32_t src_row, uint32_t dst_row, uint32_t rows) {
                            const auto src = TensorAccessor(in_args, input_addr(k), page_bytes);
                            const uint32_t src_page = src_tile_row * out_tile_cols + first_col;
                            const Piece piece(src_row, dst_row, rows);
                            const uint32_t cls =
                                parity_class(src.get_noc_addr(src_page, piece.src_off), slot_base + piece.dst_off);
                            if (cls == 0) {
                                return;
                            }
                            ASSERT(cls == stage_class);
                            for (uint32_t i = 0; i < extent; ++i) {
                                piece.template read<page_bytes>(src, src_page + i, stage_base + i * page_bytes);
                            }
                            span_lo = min_u32(span_lo, piece.dst_off);
                            span_hi = max_u32(span_hi, piece.dst_end());
                        });
                }
                if (span_hi != 0) {
                    {
                        DeviceZoneScopedN("g6c_r_stage_barrier");
                        noc_async_read_barrier();
                    }
                    {
                        DeviceZoneScopedN("g6c_merge_stage_spans_block");
                        // Stage and slot tiles share the page stride, so one copy spans every column.
                        noc_async_read(
                            get_noc_addr(stage_base + span_lo),
                            slot_base + span_lo,
                            (extent - 1) * page_bytes + span_hi - span_lo);
                    }
                    {
                        DeviceZoneScopedN("g6c_r_merge_barrier");
                        noc_async_read_barrier();
                    }
                }
            }

            {
                DeviceZoneScopedN("g6c_read_onparity_rows_block");
                for_each_run_piece(
                    num_inputs,
                    image,
                    row_lo,
                    valid_rows,
                    [&](uint32_t k, uint32_t src_tile_row, uint32_t src_row, uint32_t dst_row, uint32_t rows) {
                        const auto src = TensorAccessor(in_args, input_addr(k), page_bytes);
                        const uint32_t src_page = src_tile_row * out_tile_cols + first_col;
                        const Piece piece(src_row, dst_row, rows);
                        const uint32_t cls =
                            parity_class(src.get_noc_addr(src_page, piece.src_off), slot_base + piece.dst_off);
                        if constexpr (has_stage) {
                            if (cls != 0) {
                                return;
                            }
                        } else {
                            ASSERT(cls == 0);
                        }
                        for (uint32_t i = 0; i < extent; ++i) {
                            piece.template read<page_bytes>(src, src_page + i, slot_base + i * page_bytes);
                        }
                    });
            }
            {
                DeviceZoneScopedN("g6c_r_read_barrier");
                noc_async_read_barrier();
            }
            cb_push_back(cb_moved_pages, block_pages);
            for (uint32_t i = 0; i < extent; ++i) {
                cursor.advance();
            }
            pages_done += extent;
        }
    }
}
