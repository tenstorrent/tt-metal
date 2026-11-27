// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_concat row_gather reader (NCRISC / NOC0), for h joins whose join offsets o_k are not all
// multiples of 32. Each output page is assembled from row-range reads of the inputs whose logical
// rows land in its tile row, so no input padding row is read and no row is read twice.
//
// gather_fetch_block: the block's pages are cut into output-row segments (img, R, c_begin, len).
//   The piece table of tile row R is built once and shared by every column of the segment; the
//   loop is piece-major, page-minor so source and destination in-tile offsets stay fixed across
//   a page sweep. A piece whose (src_off - dst_off) is not a multiple of the input's NoC read
//   alignment is read into the staging mirror slot at dst_off + shift instead (shift keeps source
//   and destination congruent), everything else lands directly in the output slot.
// gather_place_block: after the fetch barrier, each staged piece is copied L1 -> L1 into its
//   output slot with a self-aimed NoC read (L1 alignment is 16 B, always satisfied).
//
// No kernel_lib helper covers interleaved row-range gathers (see op_design.md, "Helpers
// considered and rejected"); the place phase uses dataflow_kernel_lib::local_addr for its
// self-aimed source. set_read_state/read_with_state are not used because piece sizes are runtime.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_concat_dataflow_common.hpp"

namespace {

// Pieces of one output tile row cover disjoint destination row ranges inside [0, 32).
constexpr uint32_t MAX_PIECES = g6_concat::TILE_ROWS;

// A row range of input k, tile row t, that never crosses a 16-row face half on either side.
// Offsets are for face column 0; face column 1 is the same plus face_bytes on both sides.
struct Piece {
    uint32_t k;
    uint32_t in_Ht;
    uint32_t t;
    uint32_t src_off;
    uint32_t dst_off;
    uint32_t bytes;
    uint32_t shift;  // 0: direct read into the output slot; otherwise staged at dst_off + shift
};

Piece pieces[MAX_PIECES];
uint32_t num_pieces = 0;
bool table_has_staged = false;
uint32_t table_R = 0xFFFFFFFFu;

// Image, tile row and tile column of the current output page.
struct OutputRowWalk {
    uint32_t img;
    uint32_t R;
    uint32_t c;

    FORCE_INLINE void advance(uint32_t pages, uint32_t Wt, uint32_t out_Ht) {
        c += pages;
        if (c == Wt) {
            c = 0;
            if (++R == out_Ht) {
                R = 0;
                ++img;
            }
        }
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather_staging = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(4);
    constexpr bool needs_staging = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t input_l1_mask_lo = get_compile_time_arg_val(6);
    constexpr uint32_t input_l1_mask_hi = get_compile_time_arg_val(7);
    constexpr auto dram_input_args = TensorAccessorArgs<8>();
    constexpr auto l1_input_args = TensorAccessorArgs<dram_input_args.next_compile_time_args_offset()>();
    using Inputs =
        g6_concat::InputPages<decltype(dram_input_args), decltype(l1_input_args), input_l1_mask_lo, input_l1_mask_hi>;

    // Common runtime args: block_pages, K, Wt, out_Ht, N (output rows per image), then
    // (base_addr_k, n_k) per input.
    constexpr uint32_t INPUT_ARGS_BASE = 5;
    const uint32_t block_pages = get_common_arg_val<uint32_t>(0);
    const uint32_t num_inputs = get_common_arg_val<uint32_t>(1);
    const uint32_t Wt = get_common_arg_val<uint32_t>(2);
    const uint32_t out_Ht = get_common_arg_val<uint32_t>(3);
    const uint32_t out_rows = get_common_arg_val<uint32_t>(4);
    auto input_addr = [&](uint32_t k) { return get_common_arg_val<uint32_t>(INPUT_ARGS_BASE + 2 * k); };
    auto input_rows = [&](uint32_t k) { return get_common_arg_val<uint32_t>(INPUT_ARGS_BASE + 2 * k + 1); };

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    Inputs inputs(dram_input_args, l1_input_args, page_bytes);

    uint32_t staging_base = 0;
    if constexpr (needs_staging) {
        staging_base = get_write_ptr(cb_gather_staging);
        // The shift trick needs staging and output slots congruent to page bases mod the DRAM
        // read alignment: CB bases are DRAM-aligned and page_bytes is a multiple of it.
        ASSERT(staging_base % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);
    }
    static_assert(page_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0, "slot congruence needs aligned page size");

    // Piece table of output tile row R, from the join offsets alone: input k owns output rows
    // [o_k, o_k + n_k). Each input's rows inside [32R, 32R + 32) are cut at row 16 and 32 on the
    // source side and at row 16 on the destination side.
    auto build_pieces = [&](uint32_t R) {
        num_pieces = 0;
        table_has_staged = false;
        table_R = R;
        const uint32_t row0 = R * g6_concat::TILE_ROWS;
        const uint32_t row1 = row0 + g6_concat::TILE_ROWS < out_rows ? row0 + g6_concat::TILE_ROWS : out_rows;
        uint32_t o_k = 0;
        for (uint32_t k = 0; k < num_inputs && o_k < row1; ++k) {
            const uint32_t n_k = input_rows(k);
            const uint32_t a = o_k > row0 ? o_k : row0;
            const uint32_t b = o_k + n_k < row1 ? o_k + n_k : row1;
            if (a < b) {
                const uint32_t align = Inputs::read_alignment(k);
                const uint32_t in_Ht = (n_k + g6_concat::TILE_ROWS - 1) / g6_concat::TILE_ROWS;
                uint32_t src_row = a - o_k;
                uint32_t dst_row = a - row0;
                uint32_t len = b - a;
                while (len > 0) {
                    const uint32_t src_in_tile = src_row % g6_concat::TILE_ROWS;
                    const uint32_t src_room = g6_concat::FACE_ROWS - src_in_tile % g6_concat::FACE_ROWS;
                    const uint32_t dst_room = g6_concat::FACE_ROWS - dst_row % g6_concat::FACE_ROWS;
                    uint32_t rows = len < src_room ? len : src_room;
                    rows = rows < dst_room ? rows : dst_room;

                    ASSERT(num_pieces < MAX_PIECES);
                    Piece& p = pieces[num_pieces++];
                    p.k = k;
                    p.in_Ht = in_Ht;
                    p.t = src_row / g6_concat::TILE_ROWS;
                    p.src_off = (src_in_tile / g6_concat::FACE_ROWS) * 2 * face_bytes +
                                (src_in_tile % g6_concat::FACE_ROWS) * face_row_bytes;
                    p.dst_off = (dst_row / g6_concat::FACE_ROWS) * 2 * face_bytes +
                                (dst_row % g6_concat::FACE_ROWS) * face_row_bytes;
                    p.bytes = rows * face_row_bytes;
                    p.shift = (p.src_off - p.dst_off) & (align - 1);
                    ASSERT(p.shift == 0 || needs_staging);
                    ASSERT(p.shift % NOC_L1_READ_ALIGNMENT_BYTES == 0);
                    table_has_staged |= p.shift != 0;

                    src_row += rows;
                    dst_row += rows;
                    len -= rows;
                }
            }
            o_k += n_k;
        }
    };

    auto fetch_segment = [&](uint32_t img, uint32_t c_begin, uint32_t seg_pages, uint32_t slot0, uint32_t stage0) {
        for (uint32_t i = 0; i < num_pieces; ++i) {
            const Piece& p = pieces[i];
            inputs.rebase(p.k, input_addr(p.k));
            const uint32_t page0 = (img * p.in_Ht + p.t) * Wt + c_begin;
            const uint32_t dst0 = p.shift != 0 ? stage0 + p.shift : slot0;
            for (uint32_t fc = 0; fc < 2; ++fc) {
                const uint32_t src_off = p.src_off + fc * face_bytes;
                uint32_t dst = dst0 + p.dst_off + fc * face_bytes;
                for (uint32_t j = 0; j < seg_pages; ++j) {
                    noc_async_read(inputs.noc_addr(page0 + j, src_off), dst, p.bytes);
                    dst += page_bytes;
                }
            }
        }
    };

    auto place_segment = [&](uint32_t seg_pages, uint32_t slot0, uint32_t stage0) {
        Noc noc;
        UnicastEndpoint self_ep;
        for (uint32_t i = 0; i < num_pieces; ++i) {
            const Piece& p = pieces[i];
            if (p.shift == 0) {
                continue;
            }
            for (uint32_t fc = 0; fc < 2; ++fc) {
                const uint32_t off = p.dst_off + fc * face_bytes;
                uint32_t src = stage0 + off + p.shift;
                uint32_t dst = slot0 + off;
                for (uint32_t j = 0; j < seg_pages; ++j) {
                    noc.async_read(
                        self_ep,
                        CoreLocalMem<uint32_t>(dst),
                        p.bytes,
                        dataflow_kernel_lib::local_addr(src, noc.get_noc_id()),
                        {});
                    src += page_bytes;
                    dst += page_bytes;
                }
            }
        }
    };

    const uint32_t pages_per_img = out_Ht * Wt;
    OutputRowWalk walk{.img = start_page / pages_per_img, .R = (start_page % pages_per_img) / Wt, .c = start_page % Wt};

    const uint32_t num_blocks = (num_pages + block_pages - 1) / block_pages;
    uint32_t remaining = num_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Ragged last block: nominal reserve/push of block_pages, only `extent` pages assembled.
        const uint32_t extent = remaining < block_pages ? remaining : block_pages;
        {
            DeviceZoneScopedN("G6C-RESERVE");
            cb_reserve_back(cb_output_pages, block_pages);
        }
        const uint32_t slot_base = get_write_ptr(cb_output_pages);
        ASSERT(slot_base % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);
        const OutputRowWalk block_start = walk;
        bool block_has_staged = false;
        {
            DeviceZoneScopedN("G6C-FETCH");
            uint32_t left = extent;
            uint32_t offset = 0;
            while (left > 0) {
                const uint32_t seg = Wt - walk.c < left ? Wt - walk.c : left;
                if (walk.R != table_R) {
                    build_pieces(walk.R);
                }
                block_has_staged |= table_has_staged;
                fetch_segment(walk.img, walk.c, seg, slot_base + offset, staging_base + offset);
                offset += seg * page_bytes;
                left -= seg;
                walk.advance(seg, Wt, out_Ht);
            }
            noc_async_read_barrier();
        }
        if constexpr (needs_staging) {
            if (block_has_staged) {
                DeviceZoneScopedN("G6C-PLACE");
                OutputRowWalk w = block_start;
                uint32_t left = extent;
                uint32_t offset = 0;
                while (left > 0) {
                    const uint32_t seg = Wt - w.c < left ? Wt - w.c : left;
                    if (w.R != table_R) {
                        build_pieces(w.R);
                    }
                    place_segment(seg, slot_base + offset, staging_base + offset);
                    offset += seg * page_bytes;
                    left -= seg;
                    w.advance(seg, Wt, out_Ht);
                }
                noc_async_read_barrier();
            }
        }
        cb_push_back(cb_output_pages, block_pages);
        remaining -= extent;
    }
}
