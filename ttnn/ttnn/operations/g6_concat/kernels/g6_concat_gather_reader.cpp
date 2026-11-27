// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_concat row_gather reader (NCRISC / NOC0): h joins whose offsets o_k are not multiples of 32.
// Each output page is assembled in its cb_output_pages slot from row-range reads of the inputs
// whose rows land in its tile row. Output tile row R gets one run table (built once per output-row
// segment, shared by every column of the segment); each run is cut at row 16 on the source and on
// the destination so a piece never leaves a face half.
//
// Per block:
//   gather_fetch_block — every piece x face column x page: one DRAM read. Direct into the output
//                        slot when (src_off - dst_off) is DRAM-read aligned, otherwise into the
//                        cb_gather_staging mirror slot at dst_off + shift.
//   gather_place_block — every staged piece: one self-aimed L1->L1 read into the output slot.
//
// No kernel_lib helper copies runtime-sized row ranges; set_read_state<transfer_size> bakes the size
// in at compile time (local_copy_helpers_dataflow.hpp), so only its local_addr() builder is used.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_concat_common.hpp"

namespace {

using g6_concat::FACE_ROWS;
using g6_concat::TILE_ROWS;

// Every piece covers at least one destination row of a 32-row tile, so a tile row has <= 32.
constexpr uint32_t MAX_PIECES = TILE_ROWS;
constexpr uint32_t DRAM_READ_ALIGN_MASK = NOC_DRAM_READ_ALIGNMENT_BYTES - 1;

struct Piece {
    uint32_t img_pages;  // pages per image of the source input: Ht_k * Wt
    uint32_t row_page;   // first page of the source tile row inside its image: t * Wt
    uint16_t input;
    uint16_t bytes;
    uint16_t src_off;
    uint16_t dst_off;
    uint16_t shift;      // (src_off - dst_off) mod DRAM read alignment; 0 = direct read
    uint16_t face_cols;  // 2 = one transfer per face column, 1 = both face columns are contiguous
};

struct RunTable {
    Piece pieces[MAX_PIECES];
    uint32_t count = 0;
    uint32_t tile_row = 0xFFFFFFFF;
    bool any_staged = false;
};

template <uint32_t face_bytes, uint32_t row_bytes, bool use_staging>
void add_piece(
    RunTable& table, uint32_t input, uint32_t img_pages, uint32_t row_page, uint32_t s, uint32_t d, uint32_t rows) {
    Piece& p = table.pieces[table.count++];
    p.input = input;
    p.img_pages = img_pages;
    p.row_page = row_page;
    p.src_off = (s / FACE_ROWS) * 2 * face_bytes + (s % FACE_ROWS) * row_bytes;
    p.dst_off = (d / FACE_ROWS) * 2 * face_bytes + (d % FACE_ROWS) * row_bytes;
    if (rows == TILE_ROWS) {
        p.bytes = 4 * face_bytes;
        p.face_cols = 1;
    } else if (rows == FACE_ROWS && s % FACE_ROWS == 0 && d % FACE_ROWS == 0) {
        p.bytes = 2 * face_bytes;
        p.face_cols = 1;
    } else {
        p.bytes = rows * row_bytes;
        p.face_cols = 2;
    }
    p.shift = (p.src_off - p.dst_off) & DRAM_READ_ALIGN_MASK;
    if constexpr (!use_staging) {
        ASSERT(p.shift == 0);
    }
    table.any_staged |= p.shift != 0;
}

// Runs of output tile row R: input k covers output rows [o_k, o_k + n_k); its rows inside
// [32R, 32R + 32) are cut at its own tile rows and at row 16 on both sides.
template <uint32_t face_bytes, uint32_t row_bytes, bool use_staging>
void build_run_table(RunTable& table, uint32_t tile_row, uint32_t num_inputs, uint32_t sizes_arg0, uint32_t Wt) {
    table.count = 0;
    table.any_staged = false;
    table.tile_row = tile_row;
    const uint32_t y_lo = tile_row * TILE_ROWS;
    const uint32_t y_hi = y_lo + TILE_ROWS;
    uint32_t offset = 0;
    for (uint32_t k = 0; k < num_inputs && offset < y_hi; ++k) {
        const uint32_t n = get_common_arg_val<uint32_t>(sizes_arg0 + k);
        if (offset + n > y_lo) {
            const uint32_t img_pages = ((n + TILE_ROWS - 1) / TILE_ROWS) * Wt;
            uint32_t j = (y_lo > offset ? y_lo : offset) - offset;
            const uint32_t j_end = (y_hi < offset + n ? y_hi : offset + n) - offset;
            while (j < j_end) {
                const uint32_t s = j % TILE_ROWS;
                const uint32_t d = offset + j - y_lo;
                uint32_t rows = j_end - j;
                if (s == 0 && d == 0 && rows >= TILE_ROWS) {
                    rows = TILE_ROWS;
                } else {
                    if (s + rows > TILE_ROWS) {
                        rows = TILE_ROWS - s;
                    }
                    if (s < FACE_ROWS && s + rows > FACE_ROWS) {
                        rows = FACE_ROWS - s;
                    }
                    if (d < FACE_ROWS && d + rows > FACE_ROWS) {
                        rows = FACE_ROWS - d;
                    }
                }
                add_piece<face_bytes, row_bytes, use_staging>(table, k, img_pages, (j / TILE_ROWS) * Wt, s, d, rows);
                j += rows;
            }
        }
        offset += n;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather_staging = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);
    constexpr bool use_staging = get_compile_time_arg_val(5) != 0;
    constexpr auto input_args = TensorAccessorArgs<6>();

    // Common: block_pages, K, out_Wt, out_Ht, v_last, base[K], n[K].
    const uint32_t block_pages = get_common_arg_val<uint32_t>(0);
    const uint32_t num_inputs = get_common_arg_val<uint32_t>(1);
    const uint32_t Wt = get_common_arg_val<uint32_t>(2);
    const uint32_t out_Ht = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t base_arg0 = 5;
    const uint32_t sizes_arg0 = base_arg0 + num_inputs;

    uint32_t pages_left = get_arg_val<uint32_t>(0);
    uint32_t img = get_arg_val<uint32_t>(1);
    uint32_t tile_row = get_arg_val<uint32_t>(2);
    uint32_t col = get_arg_val<uint32_t>(3);

    auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(base_arg0), page_bytes);

    // Direct and staged reads rely on slot bases being DRAM-read aligned (CB bases are allocated
    // DRAM-aligned and page_bytes is a multiple of the alignment).
    ASSERT((get_write_ptr(cb_output_pages) & DRAM_READ_ALIGN_MASK) == 0);
    uint32_t staging_base = 0;
    if constexpr (use_staging) {
        staging_base = get_write_ptr(cb_gather_staging);
        ASSERT((staging_base & DRAM_READ_ALIGN_MASK) == 0);
    }

    RunTable table;

    while (pages_left > 0) {
        const uint32_t extent = pages_left < block_pages ? pages_left : block_pages;
        {
            DeviceZoneScopedN("G6C-RESERVE");
            cb_reserve_back(cb_output_pages, block_pages);
        }
        const uint32_t out_base = get_write_ptr(cb_output_pages);
        const uint32_t seg_row = tile_row, seg_col = col;
        bool block_staged = false;

        // gather_fetch_block: piece-major, page-minor inside each output-row segment.
        {
            DeviceZoneScopedN("G6C-FETCH");
            for (uint32_t i = 0; i < extent;) {
                const uint32_t seg_pages = (Wt - col < extent - i) ? Wt - col : extent - i;
                if (table.tile_row != tile_row) {
                    build_run_table<face_bytes, row_bytes, use_staging>(table, tile_row, num_inputs, sizes_arg0, Wt);
                }
                block_staged |= table.any_staged;
                const uint32_t out_slot = out_base + i * page_bytes;
                const uint32_t staging_slot = staging_base + i * page_bytes;
                for (uint32_t pi = 0; pi < table.count; ++pi) {
                    const Piece& p = table.pieces[pi];
                    input.bank_base_address = get_common_arg_val<uint32_t>(base_arg0 + p.input);
                    const uint32_t first_page = img * p.img_pages + p.row_page + col;
                    for (uint32_t fc = 0; fc < p.face_cols; ++fc) {
                        const uint32_t src_off = p.src_off + fc * face_bytes;
                        uint32_t dst = (p.shift == 0 ? out_slot : staging_slot) + p.dst_off + fc * face_bytes + p.shift;
                        for (uint32_t c = 0; c < seg_pages; ++c) {
                            noc_async_read(input.get_noc_addr(first_page + c, src_off), dst, p.bytes);
                            dst += page_bytes;
                        }
                    }
                }
                i += seg_pages;
                col += seg_pages;
                if (col == Wt) {
                    col = 0;
                    if (++tile_row == out_Ht) {
                        tile_row = 0;
                        ++img;
                    }
                }
            }
            noc_async_read_barrier();
        }

        // gather_place_block: move the staged pieces from their mirror slots into the output slots.
        if constexpr (use_staging) {
            if (block_staged) {
                DeviceZoneScopedN("G6C-PLACE");
                uint32_t p_row = seg_row, p_col = seg_col;
                for (uint32_t i = 0; i < extent;) {
                    const uint32_t seg_pages = (Wt - p_col < extent - i) ? Wt - p_col : extent - i;
                    if (table.tile_row != p_row) {
                        build_run_table<face_bytes, row_bytes, use_staging>(table, p_row, num_inputs, sizes_arg0, Wt);
                    }
                    const uint32_t out_slot = out_base + i * page_bytes;
                    const uint32_t staging_slot = staging_base + i * page_bytes;
                    for (uint32_t pi = 0; pi < table.count; ++pi) {
                        const Piece& p = table.pieces[pi];
                        if (p.shift == 0) {
                            continue;
                        }
                        for (uint32_t fc = 0; fc < p.face_cols; ++fc) {
                            const uint32_t off = p.dst_off + fc * face_bytes;
                            const auto src = dataflow_kernel_lib::local_addr(staging_slot + off + p.shift, noc_index);
                            uint64_t src_noc = get_noc_addr(src.noc_x, src.noc_y, src.addr);
                            uint32_t dst = out_slot + off;
                            for (uint32_t c = 0; c < seg_pages; ++c) {
                                noc_async_read(src_noc, dst, p.bytes);
                                src_noc += page_bytes;
                                dst += page_bytes;
                            }
                        }
                    }
                    i += seg_pages;
                    p_col += seg_pages;
                    if (p_col == Wt) {
                        p_col = 0;
                        if (++p_row == out_Ht) {
                            p_row = 0;
                        }
                    }
                }
                noc_async_read_barrier();
            }
        }

        cb_push_back(cb_output_pages, block_pages);
        pages_left -= extent;
    }
}
