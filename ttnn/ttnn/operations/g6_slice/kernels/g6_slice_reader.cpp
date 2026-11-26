// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_slice reader (NCRISC, NoC0).
//
// Helper policy: no kernel_lib helper covers a page gather from an interleaved tensor or a
// sub-tile row/column re-assembly between two tiles in L1 (op_design.md -> "Helpers considered
// and rejected"); this kernel uses the raw dataflow API, as the design's API mapping specifies.
//
// Per core: a contiguous range of `core_tiles` flattened output tiles (lead_0, lead_1,
// out_tile_row, out_tile_col of the rank-4-normalized output), walked in blocks of
// `block_tiles` (CT knob). Per block:
//   R1 (aligned begins):  load_block    -- each output tile's input page read straight into its
//                                          cb_output_tiles slot; 1 reserve + 1 barrier + 1 push.
//   R2 (SHIFT_ROWS / SHIFT_COLS): load_block -- the <= src_per_out_cap source tiles of each output
//                                          tile into fixed slots of cb_source_tiles; 1 barrier.
//                         assemble_block -- face-row runs (o_w == 0: word copies of 16-element
//                                          face rows) or element copies (o_w != 0) into the
//                                          reserved cb_output_tiles slots; 1 push.
// Every reserve/push is the nominal block_tiles (ring stays aligned); the last block's NoC
// transfers and copies cover only its real extent.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

}  // namespace

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_source_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t src_per_out_cap = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();

    // ---------------- common runtime args (patched per call) ----------------
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_lead1 = get_common_arg_val<uint32_t>(1);  // M1'
    const uint32_t out_rows = get_common_arg_val<uint32_t>(2);   // ceil(M_{r-2}/32)
    const uint32_t out_cols = get_common_arg_val<uint32_t>(3);   // ceil(M_{r-1}/32)
    const uint32_t in_lead1 = get_common_arg_val<uint32_t>(4);   // N1'
    const uint32_t ht_in = get_common_arg_val<uint32_t>(5);
    const uint32_t wt_in = get_common_arg_val<uint32_t>(6);
    const uint32_t b0 = get_common_arg_val<uint32_t>(7);
    const uint32_t b1 = get_common_arg_val<uint32_t>(8);
    const uint32_t q0 = get_common_arg_val<uint32_t>(9);   // b_{r-2} / 32
    const uint32_t c0 = get_common_arg_val<uint32_t>(10);  // b_{r-1} / 32
    [[maybe_unused]] const uint32_t o_h = get_common_arg_val<uint32_t>(11);
    [[maybe_unused]] const uint32_t o_w = get_common_arg_val<uint32_t>(12);
    [[maybe_unused]] const uint32_t m_h = get_common_arg_val<uint32_t>(13);  // M_{r-2}
    [[maybe_unused]] const uint32_t m_w = get_common_arg_val<uint32_t>(14);  // M_{r-1}

    // ---------------- per-core runtime args ----------------
    const uint32_t core_tiles = get_arg_val<uint32_t>(0);
    uint32_t i0 = get_arg_val<uint32_t>(1);
    uint32_t i1 = get_arg_val<uint32_t>(2);
    uint32_t r = get_arg_val<uint32_t>(3);
    uint32_t c = get_arg_val<uint32_t>(4);

    const auto input = TensorAccessor(input_args, in_addr, page_size);

    const uint32_t image_tiles = ht_in * wt_in;
    uint32_t image_base = ((i0 + b0) * in_lead1 + (i1 + b1)) * image_tiles;

    // Advance the 4-D output-tile counter (i0, i1, r, c) by one tile, with carry.
    auto advance = [&]() {
        if (++c == out_cols) {
            c = 0;
            if (++r == out_rows) {
                r = 0;
                if (++i1 == out_lead1) {
                    i1 = 0;
                    ++i0;
                    image_base = ((i0 + b0) * in_lead1 + b1) * image_tiles;
                } else {
                    image_base += image_tiles;
                }
            }
        }
    };

#if defined(SHIFT_ROWS) || defined(SHIFT_COLS)
#ifdef SHIFT_ROWS
    constexpr uint32_t row_span = 2;
#else
    constexpr uint32_t row_span = 1;
#endif
#ifdef SHIFT_COLS
    constexpr uint32_t col_span = 2;
#else
    constexpr uint32_t col_span = 1;
#endif
    static_assert(row_span * col_span == src_per_out_cap, "staging slots must match src_per_out_cap");
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    // cb_source_tiles is reader-private scratch (one RISC fills and reads it, ordered by the read
    // barrier): its base address is taken once and never pushed.
    const uint32_t staging_base = get_write_ptr(cb_source_tiles);
    auto rows_in_tile = [&](uint32_t tile_row) {
        const uint32_t left = m_h - tile_row * TILE_DIM;
        return left < TILE_DIM ? left : TILE_DIM;
    };
    [[maybe_unused]] auto cols_in_tile = [&](uint32_t tile_col) {
        const uint32_t left = m_w - tile_col * TILE_DIM;
        return left < TILE_DIM ? left : TILE_DIM;
    };
#endif

    const uint32_t num_blocks = (core_tiles + block_tiles - 1) / block_tiles;
    uint32_t remaining = core_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;
        remaining -= extent;

        {
            DeviceZoneScopedN("reader_reserve");
            cb_reserve_back(cb_output_tiles, block_tiles);
        }
        const uint32_t out_base = get_write_ptr(cb_output_tiles);

#if !defined(SHIFT_ROWS) && !defined(SHIFT_COLS)
        // ---------------- R1 load_block: input page -> output slot ----------------
        {
            DeviceZoneScopedN("load_block");
            uint32_t dst = out_base;
            for (uint32_t j = 0; j < extent; ++j) {
                const uint32_t in_page = image_base + (q0 + r) * wt_in + (c0 + c);
                noc_async_read_page(in_page, input, dst);
                dst += page_size;
                advance();
            }
        }
        {
            DeviceZoneScopedN("read_barrier");
            noc_async_read_barrier();
        }
#else
        // ---------------- R2 load_block: source tiles -> staging slots ----------------
        // (r, c) at the block start: assemble_block re-walks them for each tile's valid extent.
        const uint32_t block_r = r;
        const uint32_t block_c = c;
        {
            DeviceZoneScopedN("load_block");
            uint32_t slot_base = staging_base;
            for (uint32_t j = 0; j < extent; ++j) {
                const uint32_t rows_t = rows_in_tile(r);
                const uint32_t cols_t = cols_in_tile(c);
                // Only the source tiles that hold selected elements: q_b = q_a + [o_h + rows_t - 1 >= 32].
                const uint32_t dr_count = (row_span == 2 && (o_h + rows_t - 1) >= TILE_DIM) ? 2 : 1;
                const uint32_t dc_count = (col_span == 2 && (o_w + cols_t - 1) >= TILE_DIM) ? 2 : 1;
                const uint32_t row_page = image_base + (q0 + r) * wt_in + (c0 + c);
                for (uint32_t dr = 0; dr < dr_count; ++dr) {
                    for (uint32_t dc = 0; dc < dc_count; ++dc) {
                        noc_async_read_page(
                            row_page + dr * wt_in + dc, input, slot_base + (dr * col_span + dc) * page_size);
                    }
                }
                slot_base += src_per_out_cap * page_size;
                advance();
            }
        }
        {
            DeviceZoneScopedN("read_barrier");
            noc_async_read_barrier();
        }
        // ---------------- R2 assemble_block: staging -> output slots ----------------
        {
            DeviceZoneScopedN("assemble_block");
            uint32_t ar = block_r;
            uint32_t ac = block_c;
            for (uint32_t j = 0; j < extent; ++j) {
                const uint32_t src_tile = staging_base + j * src_per_out_cap * page_size;
                const uint32_t dst_tile = out_base + j * page_size;
                const uint32_t rows_t = rows_in_tile(ar);
                [[maybe_unused]] const uint32_t cols_t = cols_in_tile(ac);
                if (++ac == out_cols) {
                    ac = 0;
                    if (++ar == out_rows) {
                        ar = 0;
                    }
                }
#ifndef SHIFT_COLS
                // Row shift only: each output face row (16 elements) is one whole source face row.
                static_assert(face_row_bytes % sizeof(uint32_t) == 0, "face row must be whole words");
                constexpr uint32_t words = face_row_bytes / sizeof(uint32_t);
                for (uint32_t i = 0; i < rows_t; ++i) {
                    const uint32_t s_abs = o_h + i;
                    const uint32_t dr = s_abs >= TILE_DIM ? 1 : 0;
                    const uint32_t s = s_abs - dr * TILE_DIM;
                    const uint32_t src_row = src_tile + dr * page_size + (s >= FACE_DIM ? 2 * face_bytes : 0) +
                                             (s % FACE_DIM) * face_row_bytes;
                    const uint32_t dst_row =
                        dst_tile + (i >= FACE_DIM ? 2 * face_bytes : 0) + (i % FACE_DIM) * face_row_bytes;
                    for (uint32_t h = 0; h < 2; ++h) {
                        volatile tt_l1_ptr uint32_t* src =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_row + h * face_bytes);
                        volatile tt_l1_ptr uint32_t* dst =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_row + h * face_bytes);
                        for (uint32_t w = 0; w < words; ++w) {
                            dst[w] = src[w];
                        }
                    }
                }
#else
                // Column shift (and maybe row shift): element-granular moves.
                using elem_t = std::conditional_t<elem_bytes == 4, uint32_t, uint16_t>;
                static_assert(elem_bytes == 2 || elem_bytes == 4, "element size must be 2 or 4 bytes");
                volatile tt_l1_ptr elem_t* dst = reinterpret_cast<volatile tt_l1_ptr elem_t*>(dst_tile);
                for (uint32_t i = 0; i < rows_t; ++i) {
                    const uint32_t s_abs = o_h + i;
                    const uint32_t dr = s_abs >= TILE_DIM ? 1 : 0;
                    const uint32_t s = s_abs - dr * TILE_DIM;
                    const uint32_t src_row_off =
                        (s >= FACE_DIM ? 2 : 0) * FACE_DIM * FACE_DIM + (s % FACE_DIM) * FACE_DIM;
                    const uint32_t dst_row_off =
                        (i >= FACE_DIM ? 2 : 0) * FACE_DIM * FACE_DIM + (i % FACE_DIM) * FACE_DIM;
                    for (uint32_t jj = 0; jj < cols_t; ++jj) {
                        const uint32_t t_abs = o_w + jj;
                        const uint32_t dc = t_abs >= TILE_DIM ? 1 : 0;
                        const uint32_t t = t_abs - dc * TILE_DIM;
                        volatile tt_l1_ptr elem_t* src =
                            reinterpret_cast<volatile tt_l1_ptr elem_t*>(src_tile + (dr * col_span + dc) * page_size);
                        dst[dst_row_off + (jj >= FACE_DIM ? FACE_DIM * FACE_DIM : 0) + (jj % FACE_DIM)] =
                            src[src_row_off + (t >= FACE_DIM ? FACE_DIM * FACE_DIM : 0) + (t % FACE_DIM)];
                    }
                }
#endif
            }
        }
#endif
        cb_push_back(cb_output_tiles, block_tiles);
    }
}
